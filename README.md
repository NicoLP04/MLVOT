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

- `main.py` : le tracker (IoU + hongrois, option Kalman), les arguments, l'affichage et les exports.
- `KalmanFilter.py` : modèle mathématique du filtre de Kalman.

Le filtre est implémenté localement avec NumPy, conformément à la partie II
du sujet (§ 1), puis importé par le tracker comme demandé au § 2.1.
Son constructeur reçoit les six paramètres `dt`, `u_x`, `u_y`, `std_acc`,
`x_std_meas`, `y_std_meas`. Chaque piste initialise ensuite son propre état
`[cx, cy, 0, 0]` à partir du centroïde de sa détection. Les six paramètres
sont explicités en tête de `main.py` ; leurs valeurs numériques
(hors accélération et vitesse initiale nulles) sont des choix du projet,
car le sujet n’en impose pas.
