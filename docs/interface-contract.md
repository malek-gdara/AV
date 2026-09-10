Ce document fixe **EXACTEMENT** comment les 3 modules du pipeline communiquent entre eux.
Il doit être **lu, compris et validé** par toute l'équipe **AVANT** de coder.

> **Règle d'or** : *Tout ce qui n'est pas écrit ici est source de conflit.*


## 🏗️ Vue d'ensemble du pipeline
┌─────────────┐ Sortie 1 ┌─────────────┐ Sortie 2 ┌─────────────┐
│ Étape 1 │ ──────────────► │ Étape 2 │ ──────────────► │ Étape 3 │
│ Détection │ │ Prédiction │ │ Performance │
│ + Tracking │ │ Trajectoire │ │ + Robustesse│
└─────────────┘ └─────────────┘ └─────────────┘



**Fréquence** : 1 cycle complet par frame vidéo.

---

## 📤 Sortie Étape 1 — Détection + Tracking

### Format : objet Python `dict` (ou JSON sérialisable)

```python
{
    "frame_id": 42,                    # int
    "timestamp": 1234567890.123,       # float (secondes, epoch ou relatif)
    "objects": [                       # list
        {
            "id": 1,                   # int
            "bbox": [100.0, 200.0, 300.0, 400.0],  # list[float] x4
            "class": "person",         # str
            "conf": 0.92               # float
        },
        {
            "id": 2,
            "bbox": [500.0, 150.0, 700.0, 350.0],
            "class": "car",
            "conf": 0.88
        }
    ]
}
### Détail des champs

| Champ | Type | Obligatoire | Description |
|---|---|---|---|
| `frame_id` | `int` | ✅ | Numéro de frame (commence à 0, +1 par frame) |
| `timestamp` | `float` | ✅ | Temps en **secondes** (float) |
| `objects` | `list` | ✅ | Liste des objets. **Vide si rien détecté** |
| `objects[].id` | `int` | ✅ | ID unique tracker. Stable. `-1` si non assigné |
| `objects[].bbox` | `list[float]` | ✅ | `[x1, y1, x2, y2]` en **pixels absolus** |
| `objects[].class` | `str` | ✅ | Nom de classe (`"person"`, `"car"`, …) |
| `objects[].conf` | `float` | ✅ | Score entre `0.0` et `1.0` |

# 📤 Sortie Étape 2 — Détection + Tracking

## Format : objet Python `dict` (ou JSON sérialisable)

{
    "frame_id": 42,                    # int (MÊME que étape 1)
    "timestamp": 1234567890.125,       # float (secondes)
    "predictions": [                   # list
        {
            "id": 1,                   # int (MÊME que étape 1)
            "future_bbox": [120.0, 210.0, 320.0, 410.0],  # list[float] x4
            "horizon": 0.5,            # float (secondes dans le futur)
            "risk": "collision",       # str
            "ttc": 2.3                 # float | null
        }
    ]
}
### Détail des champs

| Champ | Type | Obligatoire | Description |
|---|---|---|---|
| `frame_id` | `int` | ✅ | Identique à l'étape 1 |
| `timestamp` | `float` | ✅ | Temps de la prédiction |
| `predictions` | `list` | ✅ | Liste. Vide si rien à prédire |
| `predictions[].id` | `int` | ✅ | Identique à l'ID de l'étape 1 |
| `predictions[].future_bbox` | `list[float]` | ✅ | Position prédite |
| `predictions[].horizon` | `float` | ✅ | Horizon en secondes (ex: 0.5) |
| `predictions[].risk` | `str` | ✅ | `none`, `collision`, `exit_zone`, `anomaly` |
| `predictions[].ttc` | `float` ou `null` | ⚠️ | Time To Collision |

# 🧪 Cas Limites (Edge Cases)

**Règle d'or** : *"Un cas limite non documenté = un bug garanti le jour J."*

Cette section liste **TOUS** les cas limites que le pipeline doit gérer. Chaque module doit s'y conformer.

---

## 📌 Catégorie 1 — Aucun objet détecté

### Cas 1.1 : Frame sans aucun objet

**Scénario** : Vidéo de nuit, scène vide, caméra qui regarde un mur.

**Comportement Étape 1** :
```python
{
    "frame_id": 42,
    "timestamp": 1.400,
    "objects": []   # ✅ liste vide, PAS null
}
**comportement etape2** :
{
    "frame_id": 42,
    "timestamp": 1.425,
    "predictions": []   # ✅ liste vide
}

**interdit**
{"objects": null}       
{"objects": None}       
{}

## Règles
- Coordonnées en PIXELS ABSOLUS (résolution 640x480)
- IDs cohérents entre étapes 1 et 2
- Pas de `null` dans les listes, utiliser des listes vides
- Timestamp en secondes (float), pas en ms
- Le lien critique : la cohérence des IDs

 CHANGEMENT DE CONTRAT v1.1
- Modification : [décrire]
- Impact : [qui est concerné]
- Raison : [pourquoi]


**Interdit** :
```python
{"objects": null}       # ❌
{"objects": None}       # ❌
{}                       # ❌ (champs manquants)
```

#### Cas 1.2 : Aucun objet pendant N frames

**Comportement** : `"objects": []` à chaque frame. Ne JAMAIS sauter une frame.

#### Cas 1.3 : Objets tous rejetés (confiance basse)

**Comportement** : `"objects": []` après filtrage. Le filtrage se fait AVANT la construction.

---

### 📌 Catégorie 2 — Objets partiels

#### Cas 2.1 : Objet sans ID

```python
{"id": -1, "bbox": [...], "class": "person", "conf": 0.65}
```

**Interdit** : `null`, `None`, `"temp"`.

#### Cas 2.2 : Objet sans classe

```python
{"id": 1, "bbox": [...], "class": "unknown", "conf": 0.50}
```

#### Cas 2.3 : Coordonnées hors image

```python
{"id": 1, "bbox": [-20.0, 150.0, 100.0, 350.0], ...}   # ✅ autorisé
```

**Règle** : ne PAS clamper. Laisser les valeurs négatives.

#### Cas 2.4 : Bbox dégénérée

**Règle** : largeur > 1 px ET hauteur > 1 px. Sinon → filtrer.

```python
def is_valid_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) > 1.0 and (y2 - y1) > 1.0
```

---

### 📌 Catégorie 3 — Tracking : problèmes d'ID

#### Cas 3.1 : ID perdu puis réassigné

**Comportement** : accepter les deux (ID réassigné ou nouvel ID). Cohérence obligatoire.

#### Cas 3.2 : Deux objets se chevauchent

**Règle** : JAMAIS 2 objets avec le même ID dans la même frame.

#### Cas 3.3 : Objet réapparaît après longue absence

**Règle** : au-delà de 30 frames d'absence → nouvel ID.

#### Cas 3.4 : Saut d'ID (bug tracker)

**Comportement** : logger un warning. Ne pas corriger dans le contrat.

---

### 📌 Catégorie 4 — Prédiction

#### Cas 4.1 : Objet sans historique

**Comportement** : exclure si < 3 frames d'historique.

#### Cas 4.2 : Objet qui sort du cadre

```python
{"risk": "exit_zone", "ttc": null}
```

#### Cas 4.3 : Collision imminente

```python
{"risk": "collision", "ttc": 0.8}   # si ttc < 2.0
```

#### Cas 4.4 : Objet immobile

**Règle** : ne PAS exclure. Inclure avec `risk: "none"`.

#### Cas 4.5 : Prediction sans ID en Étape 1

**Interdit**. Filtrer côté étape 2.

---

### 📌 Catégorie 5 — Temps et synchronisation

#### Cas 5.1 : Timestamp identique à 2 frames

**Règle** : timestamp strictement croissant.

#### Cas 5.2 : Frame sautée (drop)

**Comportement** : trous dans `frame_id` autorisés. Ne PAS renuméroter.

#### Cas 5.3 : Timestamp qui recule

**Règle** : forcer `timestamp = max(prev + 0.001, current)`.

#### Cas 5.4 : Latence négative

**Règle** : impossible. Logger warning et ignorer.

---

### 📌 Catégorie 6 — Erreurs et exceptions

#### Cas 6.1 : Inférence échoue

```python
try:
    results = model(frame)
except Exception as e:
    print(f"[ERROR] Inference failed: {e}")
    return {"frame_id": frame_id, "timestamp": timestamp, "objects": []}
```

**Règle** : AUCUNE exception ne doit sortir d'un module.

#### Cas 6.2 : Frame corrompue

**Comportement** : skip + log.

#### Cas 6.3 : Modèle non chargé

**Comportement** : crash immédiat au démarrage.

#### Cas 6.4 : Liste modifiée pendant itération

**Interdit**. Toujours créer une nouvelle liste.

---

### 📌 Catégorie 7 — Cas numériques

#### Cas 7.1 : conf = 0.0 ou 1.0

- `conf = 0.0` → filtrer
- `conf = 1.0` → garder

**Règle** : `0.0 < conf <= 1.0`.

#### Cas 7.2 : NaN ou Inf

**Règle** : filtrer. Jamais propager.

```python
import math
def is_valid_number(x):
    return isinstance(x, (int, float)) and math.isfinite(x)
```

#### Cas 7.3 : Coordonnées très grandes

**Règle** : filtrer si bbox > 2× taille image.

---

### 📌 Catégorie 8 — Mode dégradé

#### Cas 8.1 : FPS trop bas

```python
if measured_fps < target_fps * 0.5:
    skip_frames = 2
    imgsz = 416
```

#### Cas 8.2 : Latence étape 2 > budget

**Comportement** : sortir `"predictions": []`, logger warning.

#### Cas 8.3 : Skip de frame intentionnel

**Règle** : `frame_id` reflète la vraie frame. Les trous sont OK.

---

### 📌 Catégorie 9 — Intégration

#### Cas 9.1 : Décalage de frames entre étapes

**Interdit**. Bufferiser si nécessaire.

#### Cas 9.2 : Champ manquant

**Comportement** : valeur par défaut + warning.

#### Cas 9.3 : Type incorrect

```python
def normalize_bbox(bbox):
    if isinstance(bbox, str):
        return [float(x) for x in bbox.split(",")]
    if isinstance(bbox, (list, tuple)):
        return [float(x) for x in bbox]
    raise ValueError(f"Invalid bbox type: {type(bbox)}")
```

**Règle** : tolérance max côté consommateur, rigueur max côté producteur.

