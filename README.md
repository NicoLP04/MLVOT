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

## PyCharm

Utiliser `.venv/bin/python` comme interpréteur du projet et `main.py`
comme script, avec le dossier du dépôt comme répertoire de travail.

Pour ajouter une dépendance au projet : `uv add nom-du-paquet`.
Versionner `pyproject.toml`, `.python-version` et `uv.lock` pour partager
le même environnement.

## Organisation du code

- `main.py` : point d’entrée, arguments, lecture des détections, affichage et exports.
- `tracker_iou.py` : suivi IOU, association hongroise et dessin des pistes.
- `tracker_kalman.py` : suivi avec prédiction Kalman et dessin des prédictions ;
  réutilise les primitives IOU pour l’association.
- `KalmanFilter.py` : modèle mathématique du filtre de Kalman.
