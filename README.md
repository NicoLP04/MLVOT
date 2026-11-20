# MLVOT — tracker IoU

## Installation avec uv

Depuis la racine du dépôt, avec [uv](https://docs.astral.sh/uv/) installé :

```bash
uv sync --locked
cp .env.example .env
```

`uv` installe la version de Python indiquée dans `.python-version`, crée
l'environnement `.venv` et installe les dépendances fixées par `uv.lock`.
Le fichier `.env` contient les variables locales ; il est ignoré par Git.
Actuellement, il active uniquement l'affichage immédiat des logs.
Les paramètres du tracker se passent en arguments de commande.

## Lancer le tracker

```bash
uv run --env-file .env main.py
```

Par défaut : détections `Yolov5l`, export MOT dans `ADL-Rundle-6.txt`.
Le mode par défaut est IOU. Pour utiliser IOU avec la prédiction Kalman :

```bash
uv run --env-file .env main.py --tracker kalman
```

Les options `--show`, `--video`, `--out` et la source de détections sont
disponibles dans les deux modes (`--tracker iou` ou `--tracker kalman`).

Aucune activation manuelle de `.venv` n'est nécessaire.

Afficher les images (touche `q` pour quitter) et enregistrer la vidéo :

```bash
uv run --env-file .env main.py Yolov5l --show --video tracking.mp4
```

Autres sources de détections :

```bash
uv run --env-file .env main.py Yolov5s --out ADL-Rundle-6.txt
uv run --env-file .env main.py public-dataset --out ADL-Rundle-6.txt
```

`--show` nécessite une session graphique. L'export texte et `--video` peuvent
s'exécuter sans fenêtre. Les données sont lues dans `ADL-Rundle-6/` à côté du script.
Les dossiers parents des chemins `--out` et `--video` doivent déjà exister.

## Évaluation (Partie III)

TrackEval n'est pas versionné ; le cloner à la racine du dépôt :

```bash
git clone https://github.com/JonathonLuiten/TrackEval.git
uv run evaluate.py   # HOTA, IDF1, IDSW, FPS des deux trackers -> results/results.md
uv run tune.py       # recherche d'hyperparamètres -> results/tuning.md (plusieurs minutes)
```

Les images de comparaison IoU / Kalman sont dans `figures/`.

### Résultats

Séquence ADL-Rundle-6, mêmes détections pour les deux trackers. Réglages par
défaut : `IOU_THRESHOLD = 0.3`, `MAX_MISSED_FRAMES = 5`, `STD_ACC = 1`,
`STD_MEAS = 5`. Le FPS ne mesure que le tracking (sans lecture d'images ni
affichage), meilleur de 3 essais.

| Détections | Tracker | HOTA | IDF1 | IDSW | FPS |
|---|---|---|---|---|---|
| Yolov5s | IoU | 39.83 | 49.58 | 64 | 11628 |
| Yolov5s | Kalman | 41.26 | 51.70 | 63 | 2197 |
| Yolov5l | IoU | 41.69 | 51.57 | 59 | 8195 |
| Yolov5l | Kalman | 42.72 | 52.91 | 52 | 1740 |
| public-dataset | IoU | 19.15 | 22.72 | 203 | 5241 |
| public-dataset | Kalman | 20.94 | 25.06 | 183 | 1277 |

Kalman fait un peu mieux que l'IoU seul (+1 à +1.8 point de HOTA, moins de
changements d'identité), mais environ 5 fois plus lent (largement au-dessus du
temps réel). Les détections pèsent plus que le tracker : avec public-dataset,
le HOTA est deux fois plus bas.

**Réglage des hyperparamètres** (`tune.py`, grille réglée sur Yolov5l, puis
vérifiée sur les deux autres détections qui n'ont pas servi au choix) :

| Détections | Tracker | Paramètres | HOTA | IDF1 | IDSW |
|---|---|---|---|---|---|
| Yolov5l (réglage) | IoU | défaut | 41.69 | 51.57 | 59 |
| Yolov5l (réglage) | IoU | `iou>=0.5` | 41.81 | 51.41 | 63 |
| Yolov5l (réglage) | Kalman | défaut | 42.72 | 52.91 | 52 |
| Yolov5l (réglage) | Kalman | `iou>=0.2, max_missed=30, std_acc=0.5, std_meas=10` | 44.62 | 57.67 | 41 |
| Yolov5s (vérif.) | Kalman | défaut | 41.26 | 51.70 | 63 |
| Yolov5s (vérif.) | Kalman | réglé | 43.97 | 59.53 | 61 |
| public-dataset (vérif.) | Kalman | défaut | 20.94 | 25.06 | 183 |
| public-dataset (vérif.) | Kalman | réglé | 23.71 | 30.83 | 183 |

- IoU seul : le réglage n'apporte rien (écart de bruit, et moins bon sur les
  autres détections). Les valeurs par défaut sont conservées.
- Kalman : le gain se retrouve sur les détections de vérification. Il profite
  d'une mémoire plus longue (`max_missed` 20-30), ce que l'IoU seul ne fait pas.
- Les valeurs réglées de Kalman ne sont pas encore appliquées dans `main.py`.
  Plusieurs optimums sont au bord de la grille (`max_missed=30`, `std_acc=0.5`,
  `std_meas=10`) : l'optimum réel peut être plus loin.

**Analyse des échecs** (détections Yolov5l, comparaison au gt, voir `figures/`) :

- Seulement 71 % des boîtes du gt sont retrouvées par le détecteur : c'est la
  limite principale, identique pour les deux trackers.
- Environ 3/4 des changements d'identité arrivent après une disparition de plus
  de 5 frames (`MAX_MISSED_FRAMES`) : la track est déjà supprimée, aucun des deux
  trackers ne peut garder l'identité (exemple : `figures/B_both_switch_gap35_f141.png`).
- Le mouvement rapide n'explique pas les échecs dans cette séquence.
- Un seul run sur une seule séquence : des écarts de 1 à 2 points de HOTA ne sont
  pas forcément significatifs.

## PyCharm

Utiliser `.venv/bin/python` comme interpréteur du projet et `main.py`
comme script, avec le dossier du dépôt comme répertoire de travail.

Pour ajouter une dépendance au projet : `uv add nom-du-paquet`.
Versionner `pyproject.toml`, `.python-version` et `uv.lock` pour partager
le même environnement.

## Organisation du code

- `main.py` : le tracker (IoU + hongrois, option Kalman), les arguments, l'affichage et les exports.
- `KalmanFilter.py` : modèle mathématique du filtre de Kalman.
- `evaluate.py`, `tune.py` : évaluation avec TrackEval et recherche d'hyperparamètres.

Le filtre est implémenté localement avec NumPy, conformément à la partie II
du sujet (§ 1), puis importé par le tracker comme demandé au § 2.1.
Son constructeur reçoit les six paramètres `dt`, `u_x`, `u_y`, `std_acc`,
`x_std_meas`, `y_std_meas`. Chaque piste initialise ensuite son propre état
`[cx, cy, 0, 0]` à partir du centroïde de sa détection. Les six paramètres
sont explicités en tête de `main.py` ; leurs valeurs numériques
(hors accélération et vitesse initiale nulles) sont des choix du projet,
car le sujet n’en impose pas.
