"""
trainer.py
----------
Classe `LSTMTrainer` responsable du cycle de vie complet de l'entraînement
d'un modèle récurrent de prédiction de trajectoires :
    - construction du modèle, de l'optimiseur, de la loss et du scheduler
    - boucle d'entraînement (mini-batches, shuffle)
    - boucle de validation
    - sauvegarde / chargement de checkpoints
    - gestion automatique GPU / CPU

Cette classe ne contient aucune logique d'inférence "métier" (cela est du
ressort de `TrajectoryPredictor`), ce qui garantit une séparation claire
des responsabilités.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models import TrajectoryRNN

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class LSTMTrainer:
    """
    Encapsule l'entraînement et la validation d'un modèle `TrajectoryRNN`.

    Parameters
    ----------
    input_size : int
        Nombre de features en entrée (ex: 2 pour (x, y), 4 pour (x, y, vx, vy)).
    hidden_size : int
        Taille de l'état caché du LSTM/GRU.
    num_layers : int
        Nombre de couches récurrentes.
    output_size : int
        Nombre de features en sortie.
    rnn_type : str, optional
        "lstm" (par défaut) ou "gru" — permet de faire évoluer l'architecture
        sans changer l'API du Trainer.
    learning_rate : float, optional
        Taux d'apprentissage initial pour Adam (par défaut 1e-3).
    device : str, optional
        "cuda", "cpu" ou None (auto-détection, par défaut).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        rnn_type: str = "lstm",
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        # --- Gestion automatique GPU / CPU -------------------------------
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info("LSTMTrainer utilise le device : %s", self.device)

        # --- Modèle --------------------------------------------------------
        self.model = TrajectoryRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            rnn_type=rnn_type,
        ).to(self.device)

        # --- Optimiseur, loss, scheduler ------------------------------------
        self.optimizer: Optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        # Réduit le LR si la loss de validation stagne : utile pour la convergence fine.
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Historique des métriques (utile pour tracer les courbes de loss).
        self.history = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------ #
    # Entraînement
    # ------------------------------------------------------------------ #
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        verbose: bool = True,
    ) -> dict:
        """
        Entraîne le modèle sur `epochs` époques.

        Parameters
        ----------
        train_loader : DataLoader
            DataLoader fournissant des batches (X, y).
        val_loader : DataLoader, optional
            DataLoader de validation. Si fourni, la loss de validation est
            calculée à chaque époque et utilisée par le scheduler.
        epochs : int
            Nombre d'époques d'entraînement.
        verbose : bool
            Affiche les métriques à chaque époque si True.

        Returns
        -------
        dict
            Historique des losses train/val.
        """
        for epoch in range(1, epochs + 1):
            train_loss = self._train_one_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            val_loss = None
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                self.history["val_loss"].append(val_loss)
                self.scheduler.step(val_loss)

            if verbose:
                msg = f"Époque {epoch}/{epochs} - train_loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - val_loss: {val_loss:.6f}"
                current_lr = self.optimizer.param_groups[0]["lr"]
                msg += f" - lr: {current_lr:.2e}"
                logger.info(msg)

        return self.history

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        """Effectue une passe d'entraînement complète (une époque)."""
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()
            self.optimizer.step()

            batch_size = batch_x.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return running_loss / max(n_samples, 1)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> float:
        """
        Calcule la loss moyenne (MSE) sur un jeu de validation/test.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader fournissant des batches (X, y).

        Returns
        -------
        float
            Loss moyenne sur l'ensemble du dataset.
        """
        self.model.eval()
        running_loss = 0.0
        n_samples = 0

        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)

            batch_size = batch_x.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

        return running_loss / max(n_samples, 1)

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde le modèle, l'optimiseur et la configuration dans un
        unique fichier checkpoint, afin de permettre un rechargement complet
        (reprise d'entraînement ou inférence pure).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
 
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": self.model.get_config(),
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info("Modèle sauvegardé dans : %s", path)

    