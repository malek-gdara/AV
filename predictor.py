"""
predictor.py
------------
Classe `TrajectoryPredictor` dédiée à l'inférence : chargement d'un modèle
entraîné (produit par `LSTMTrainer`) et prédiction de trajectoires en
temps réel.

Inclut :
    - Linear (baseline rapide)
    - Kalman simplifié (robustesse face au bruit)
    - LSTM (IA avancée, modèle entraîné)
    - Détection d’anomalies (distance + variance)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from trainer import LSTMTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


class TrajectoryPredictor:
    """
    Interface d'inférence temps réel pour la prédiction de trajectoires.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        anomaly_threshold: float = 1.0,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Chargement du modèle LSTM uniquement (inférence pure).
        self.model = LSTMTrainer.load_model_only(model_path, device=str(self.device))
        self.model.eval()

        self.input_size = self.model.input_size
        self.output_size = self.model.output_size
        self.anomaly_threshold = anomaly_threshold

        logger.info(
            "TrajectoryPredictor prêt (device=%s, input_size=%d, output_size=%d)",
            self.device, self.input_size, self.output_size,
        )

    # ------------------------------------------------------------------ #
    # Méthode 1 : Linear prediction
    # ------------------------------------------------------------------ #
    def linear_predict(self, positions: List[tuple], steps: int = 5):
        """
        Extrapolation linéaire basée sur la dernière vitesse.
        positions : [(x,y,vx,vy), ...]
        """
        if len(positions) < 2:
            return positions, 0.0

        (x2, y2, vx2, vy2) = positions[-1]
        preds = [(x2 + vx2*i, y2 + vy2*i, vx2, vy2) for i in range(1, steps+1)]
        risk = self._risk_score(preds)
        return preds, risk

    # ------------------------------------------------------------------ #
    # Méthode 2 : Kalman simplifié
    # ------------------------------------------------------------------ #
    def kalman_predict(self, positions: List[tuple], steps: int = 5):
        """
        Filtre simplifié : moyenne glissante des positions et vitesses.
        """
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        vxs = [p[2] for p in positions]
        vys = [p[3] for p in positions]

        mean_x, mean_y = np.mean(xs), np.mean(ys)
        mean_vx, mean_vy = np.mean(vxs), np.mean(vys)

        preds = [(mean_x + mean_vx*i, mean_y + mean_vy*i, mean_vx, mean_vy) for i in range(1, steps+1)]
        risk = self._risk_score(preds)
        return preds, risk

    # ------------------------------------------------------------------ #
    # Méthode 3 : LSTM prediction
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def lstm_predict(self, sequence: ArrayLike, steps: int = 1) -> np.ndarray:
        """
        LSTM entraîné : capture des patterns complexes.
        """
        tensor_seq = self._to_tensor(sequence).unsqueeze(0)  # (1, seq_len, input_size)

        predictions = []
        current_seq = tensor_seq

        for _ in range(steps):
            pred = self.model(current_seq)  # (1, output_size)
            predictions.append(pred.squeeze(0).cpu().numpy())

            # Réinjecte la prédiction avec recalcul de vitesse
            next_input = self._prepare_next_input(pred, current_seq[:, -1, :])
            current_seq = torch.cat([current_seq[:, 1:, :], next_input], dim=1)

        return np.stack(predictions, axis=0)

    # ------------------------------------------------------------------ #
    # Détection d'anomalies
    # ------------------------------------------------------------------ #
    def detect_anomaly(self, predicted_position: ArrayLike, real_position: ArrayLike) -> bool:
        pred = np.asarray(predicted_position, dtype=np.float32)
        real = np.asarray(real_position, dtype=np.float32)
        distance = float(np.linalg.norm(pred[:2] - real[:2]))
        return distance > self.anomaly_threshold

    def anomaly_score(self, positions: List[tuple]) -> float:
        coords = np.array([[x,y] for (x,y,vx,vy) in positions])
        var = np.var(coords, axis=0).mean()
        return var

    # ------------------------------------------------------------------ #
    # Risk score combiné
    # ------------------------------------------------------------------ #
    def _risk_score(self, preds: List[tuple]) -> float:
        risk = 0.0
        for i in range(len(preds)-1):
            dx = preds[i+1][0] - preds[i][0]
            dy = preds[i+1][1] - preds[i][1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 1.0:
                risk += 0.5

        anomaly_risk = self.anomaly_score(preds)
        return min(1.0, risk + anomaly_risk)

    # ------------------------------------------------------------------ #
    # Utilitaires internes
    # ------------------------------------------------------------------ #
    def _to_tensor(self, sequence: ArrayLike) -> torch.Tensor:
        arr = np.asarray(sequence, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != self.input_size:
            raise ValueError(
                f"Séquence de forme invalide {arr.shape}, "
                f"attendu (seq_len, {self.input_size})."
            )
        return torch.from_numpy(arr).to(self.device)

    def _prepare_next_input(self, prediction: torch.Tensor, last_input: torch.Tensor) -> torch.Tensor:
        """
        Réinjecte la prédiction en recalculant la vitesse si nécessaire.
        """
        batch_size = prediction.size(0)
        if self.output_size == self.input_size:
            next_step = prediction
        else:
            # Recalcule vx, vy à partir de la différence de positions
            px, py = prediction[0, 0].item(), prediction[0, 1].item()
            lx, ly = last_input[0, 0].item(), last_input[0, 1].item()
            vx, vy = px - lx, py - ly
            next_step = torch.tensor([[px, py, vx, vy]], device=prediction.device)
        return next_step.unsqueeze(1)  # (batch, 1, input_size)
