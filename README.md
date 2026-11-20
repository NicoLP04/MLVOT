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

## Tracker avec apparence (Partie IV)

```bash
uv run main.py --tracker appearance --descriptor hist   # histogramme de couleurs (naïf)
uv run main.py --tracker appearance --descriptor cnn    # MobileNetV3 pré-entraîné
uv run main.py --tracker appearance --descriptor reid   # modèle de ré-identification YOLO26m-ReID (défaut)
```

Les descripteurs des détections sont calculés une fois puis mis en cache dans
`results/features/`. Le mode `cnn` télécharge les poids de MobileNetV3 au premier
lancement et nécessite PyTorch (version CPU, déclarée dans `pyproject.toml`).

Détections d'un détecteur récent (YOLO26x, COCO, classe « person »), générées hors
ligne puis lues comme les autres (environ 10 min sur CPU, poids téléchargés au
premier lancement) :

```bash
uv run detect.py                                   # -> ADL-Rundle-6/det/yolo26x/det.txt (conf >= 0.7)
uv run main.py yolo26x --tracker kalman --out kalman_y26.txt
uv run main.py yolo26x --tracker appearance --out app_y26.txt      # meilleur tracker : HOTA 49.34
```

`--interpolate` remplit les frames où une track n'a pas été vue (post-traitement
hors ligne, les boîtes ajoutées ne sont pas dessinées dans la vidéo).

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

| Détections | Tracker | HOTA | DetA | AssA | IDF1 | IDSW | FPS |
|---|---|---|---|---|---|---|---|
| Yolov5s | IoU | 39.83 | 44.87 | 36.28 | 49.58 | 64 | 12679 |
| Yolov5s | Kalman | 41.26 | 44.91 | 38.83 | 51.70 | 63 | 2286 |
| Yolov5l | IoU | 41.69 | 46.65 | 38.09 | 51.57 | 59 | 8834 |
| Yolov5l | Kalman | 42.72 | 47.08 | 39.56 | 52.91 | 52 | 1761 |
| public-dataset | IoU | 19.15 | 31.45 | 11.92 | 22.72 | 203 | 5158 |
| public-dataset | Kalman | 20.94 | 31.33 | 14.42 | 25.06 | 183 | 1281 |

DetA mesure la qualité de détection, AssA la qualité d'association des
identités. L'AssA est plus bas que la DetA partout (très net avec
public-dataset) : l'association est le point faible. Kalman n'agit que sur
l'AssA (la DetA ne bouge pas). Le FPS varie d'environ 10 % d'un lancement à
l'autre.

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

### Partie IV : tracker avec apparence

**Idée** : les limites de la Partie III sont surtout des disparitions longues
(l'association, AssA, est le point faible). Le tracker Kalman est étendu ainsi :
la similarité devient `S = α·IoU + β·cos(apparence)` (formule du sujet), chaque
track garde un descripteur d'apparence (moyenne mobile), et les tracks perdues
sont conservées plus longtemps (30 frames) pour être ré-associées à leur
réapparition. Deux descripteurs, avec la même procédure : un histogramme de
couleurs HSV (haut et bas du corps, naïf) et MobileNetV3-small ImageNet (CNN).
Réglage sur Yolov5l uniquement (`tune_appearance.py`, 40 configurations :
β, seuil de S, mémoire), meilleure configuration identique pour les deux :
β = 0.5, seuil 0.4, mémoire 30. Les autres détections servent de vérification.

| Détections | Tracker | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| Yolov5l (réglage) | Kalman (Partie II, sujet) | 42.72 | 47.08 | 39.56 | 52.91 | 52 |
| Yolov5l (réglage) | Kalman réglé | 44.62 | 46.78 | 43.21 | 57.67 | 41 |
| Yolov5l (réglage) | Apparence, histogramme | 45.14 | 46.76 | 44.22 | 60.12 | 37 |
| Yolov5l (réglage) | Apparence, CNN | **46.05** | 46.66 | 46.02 | 61.14 | 28 |
| Yolov5s (vérif.) | Kalman (Partie II, sujet) | 41.26 | 44.91 | 38.83 | 51.70 | 63 |
| Yolov5s (vérif.) | Kalman réglé | 43.97 | 44.62 | 43.99 | 59.53 | 61 |
| Yolov5s (vérif.) | Apparence, histogramme | **44.36** | 44.58 | 44.92 | 61.04 | 39 |
| Yolov5s (vérif.) | Apparence, CNN | 42.05 | 43.88 | 41.18 | 54.89 | 37 |
| public-dataset (vérif.) | Kalman (Partie II, sujet) | 20.94 | 31.33 | 14.42 | 25.06 | 183 |
| public-dataset (vérif.) | Kalman réglé | 23.71 | 31.01 | 18.48 | 30.83 | 183 |
| public-dataset (vérif.) | Apparence, histogramme | **24.21** | 31.26 | 19.00 | 32.39 | 161 |
| public-dataset (vérif.) | Apparence, CNN | 22.81 | 30.58 | 17.25 | 28.78 | 181 |

**Lecture**

- Par rapport au Kalman du sujet, les deux versions gagnent sur le réglage
  (+2.4 et +3.3 points de HOTA avec Yolov5l), mais **la plus grande partie vient
  du réglage de Kalman lui-même** (mémoire plus longue, bruit). La contribution
  propre de l'apparence est la comparaison avec « Kalman réglé » : de +0.4 à +0.5
  point de HOTA avec l'histogramme, ce qui est petit.
- **Le descripteur CNN ne se généralise pas** : meilleur sur Yolov5l (où on a
  réglé), mais moins bon que Kalman réglé sur les deux autres détections
  (HOTA 42.05 contre 43.97, et 22.81 contre 23.71). L'histogramme, plus simple,
  est meilleur sur toutes les détections de vérification. Hypothèse non
  vérifiée : MobileNetV3 est entraîné sur ImageNet (classification d'objets),
  pas pour distinguer des personnes ; un modèle de ré-identification (OSNet)
  ferait probablement mieux, mais n'a pas été testé.
- **Les deux réduisent les changements d'identité** (IDSW) sur les trois
  détections, par exemple 61 → 39 (histogramme) et 61 → 37 (CNN) avec Yolov5s.
  Sur Yolov5l, d'après notre comparaison au gt, le gain se concentre sur les
  disparitions de 6 à 30 frames. Au-delà de 30 frames (11 cas), rien ne change :
  la track est supprimée.
- **Non résolu** : la détection (DetA reste autour de 47 avec YOLO et 31 avec
  public-dataset). Aucun de ces trackers n'améliore les personnes jamais détectées.
- **Vitesse** (FPS de tracking seul, Yolov5l) : environ 1230-1300 FPS, proche de
  Kalman réglé (1500), mais le calcul des descripteurs est à part : le run complet
  de `main.py` prend environ 11 s (histogramme) et 74 s (CNN) sur 525 frames sans
  cache, soit environ 45 et 7 images par seconde sur 12 cœurs CPU. Le CNN est donc
  sous le temps réel.
- Un seul run, une seule séquence : les écarts de moins d'un point de HOTA ne sont
  pas forcément significatifs.

**Visualisations** (Yolov5l, `figures/C_*` et `figures/D_*`) : une ligne par
tracker (Kalman Partie II, Kalman réglé, apparence histogramme, apparence CNN),
trois colonnes (dernière frame où la personne est vue, milieu de la disparition,
réapparition). Le contour blanc fin est la vérité terrain de la personne suivie.

- `C_appearance_recovers_gap16_f133.png` : une personne cachée 16 frames derrière
  un homme. Les deux Kalman lui donnent un nouvel identifiant (6 → 23, 6 → 18),
  les deux trackers avec apparence gardent 6.
- `C_appearance_recovers_gap28_f171.png` : une petite fille cachée 28 frames
  derrière un adulte. Kalman réglé a pourtant aussi une mémoire de 30 frames,
  mais change d'identifiant (21 → 26) : la track 21 existe encore à la frame 171,
  mais après 26 frames sans correction sa boîte prédite a dérivé (x = 1312 au lieu
  de 1344 pour la détection) et l'IoU vaut 0.10, sous le seuil de 0.2. Avec
  l'apparence, S = 0.5·IoU + 0.5·cos passe le seuil grâce au descripteur : les
  deux gardent l'identifiant (20 et 19).
- `D_all_switch_gap35_f141.png` : la même fille, plus tôt, cachée 35 frames.
  Les quatre trackers lui donnent un nouvel identifiant : la disparition dépasse
  la mémoire de 30 frames, la track est supprimée avant la réapparition.
- `D_all_swap_overlap_f102.png` : la fille est devant une autre personne ; à la
  frame 102 les quatre trackers mettent un autre identifiant sur elle pendant une
  frame, puis reviennent à 3. L'apparence n'empêche pas ce changement bref.

### Partie IV : meilleur détecteur et interpolation

**Détecteur YOLO26x** (`detect.py`, pré-entraîné COCO, images en 1280 px). Le
seuil de confiance a été choisi sur cette séquence (0.1 à 0.8 testés) : 0.7.

| Détections | Tracker | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|---|
| Yolov5l | Kalman (Partie II) | 42.72 | 47.08 | 39.56 | 52.91 | 52 |
| Yolov5l | Kalman réglé | 44.62 | 46.78 | 43.21 | 57.67 | 41 |
| Yolov5l | Apparence CNN | 46.05 | 46.66 | 46.02 | 61.14 | 28 |
| YOLO26x | Kalman (Partie II) | 44.21 | 50.18 | 39.65 | 55.29 | 57 |
| YOLO26x | Kalman réglé | **46.77** | 50.19 | 44.08 | 62.54 | 38 |
| YOLO26x | Apparence histogramme | 46.52 | 49.65 | 44.10 | 62.03 | 39 |
| YOLO26x | Apparence CNN | 45.20 | 49.52 | 41.82 | 57.71 | 30 |

- Gain de +1.5 point de HOTA (Kalman Partie II) à +2.2 (Kalman réglé), venant du
  DetA (+3). Avec YOLO26x, l'apparence n'apporte plus rien par rapport à Kalman réglé.
- **Pourquoi le gain est faible** : en comparant les détections au gt, 94 % des
  boîtes du gt qu'aucun détecteur ne trouve sont couvertes à plus de 50 % par une
  autre personne (82 % à plus de 80 %) : ce sont des personnes cachées, que le gt
  annote quand même. Aucun détecteur ne peut les voir. À l'inverse, une bonne
  partie des « faux positifs » sont de vraies personnes non annotées (au fond, dans
  les magasins). Filtrer les boîtes de moins de 120 px de haut donne +0.4 HOTA, mais
  c'est un réglage sur la façon d'annoter cette séquence, il n'est pas gardé.

**Interpolation des trous** (`--interpolate`) : idée testée pour retrouver les
personnes cachées, quasiment sans effet (de -0.3 à +0.4 HOTA selon le tracker,
le DetA baisse). Sur Yolov5l avec l'apparence CNN, seulement 23 % des boîtes
interpolées sur des trous de 16 à 30 frames tombent sur une personne du gt : la
plupart des trous appartiennent à des tracks sur des personnes non annotées (les
deux extrémités du trou ne sont sur le gt que dans 15 % des cas). Les boîtes
ajoutées sont alors des faux positifs, qui compensent les vraies personnes cachées
retrouvées.

### Partie IV : vrai modèle de ré-identification (ReID)

Le CNN ImageNet ne se généralisait pas. Il est remplacé par un modèle entraîné à
distinguer des personnes : YOLO26m-ReID d'Ultralytics (entraîné sur MSMT17, 1041
identités, entrée 448×448, vecteur de 512), lu avec `onnxruntime`
(`--descriptor reid`). Même protocole : grille de 40 configurations sur Yolov5l,
meilleure β = 0.5, seuil 0.3, mémoire 30 (seuil au bord bas de la grille).

| Détections | Kalman réglé | Apparence histogramme | Apparence CNN | Apparence ReID |
|---|---|---|---|---|
| Yolov5l (réglage), HOTA (IDSW) | 44.62 (41) | 45.14 (37) | 46.05 (28) | **47.11** (36) |
| Yolov5s (vérif.) | 43.97 (61) | 44.36 (39) | 42.05 (37) | **44.91** (34) |
| public-dataset (vérif.) | 23.71 (183) | 24.21 (161) | 22.81 (181) | **26.06** (143) |
| YOLO26x | 46.77 (38) | 46.52 (39) | - | **46.82** (34) |
| YOLO26x + `--interpolate` | 46.90 (34) | 46.58 (35) | - | **47.28** (29) |

- Le ReID est le meilleur descripteur sur toutes les détections, y compris
  celles qui n'ont pas servi au réglage (+0.9 à +2.4 HOTA par rapport à Kalman
  réglé, le gain vient de l'AssA). Contrairement au CNN ImageNet, il se généralise.
- Avec YOLO26x, le gain du ReID devient négligeable (+0.05 sans interpolation,
  +0.4 avec) : les gains du détecteur et de l'apparence ne s'additionnent pas.
  Les paramètres ont été réglés sur Yolov5l, pas sur YOLO26x.
- Coût : environ 150 ms par boîte sur CPU (environ 10 min de descripteurs pour
  525 frames), très loin du temps réel. Le tracking seul reste à environ 1600 FPS.

**Réglage sur YOLO26x** (216 configurations : β, seuil de S, mémoire, `std_acc`,
`std_meas`). Meilleure : β = 0.25, seuil 0.15, mémoire 60, `std_acc` 0.25,
`std_meas` 10 → **49.34 HOTA** (DetA 49.64, AssA 49.44, IDF1 67.40, IDSW 28), contre
46.82 avec les paramètres réglés sur Yolov5l. Plusieurs configurations voisines
donnent 49.2-49.3 : ce n'est pas un point isolé.

- **Ces paramètres ne se transfèrent pas** : sur Yolov5l, Yolov5s et
  public-dataset, ils font moins bien que les paramètres réglés sur Yolov5l
  (46.71 contre 47.11, 44.76 contre 44.91, 24.32 contre 26.06). Les bons paramètres
  dépendent du détecteur : avec les détections plus propres de YOLO26x, l'IoU
  peut avoir plus de poids (β = 0.25) et le seuil être plus bas.
- Le score de 49.34 est réglé sur la vérité terrain de la séquence évaluée : il est
  optimiste. L'optimum est encore au bord de la grille (β, seuil, mémoire).
- L'interpolation fait baisser ce score (48.79) : avec une mémoire de 60 frames,
  les trous interpolés sont plus longs et plus souvent faux.
- Effet moyen de chaque paramètre sur la grille : β et le seuil comptent le plus
  (HOTA moyenne 47.7 / 45.2 / 41.3 pour β = 0.25 / 0.5 / 0.75), la mémoire et le
  bruit de Kalman beaucoup moins (moins de 0.4 point d'écart moyen).
- Ces valeurs sont les valeurs par défaut de `--tracker appearance` dans `main.py`.
- Meilleur score obtenu : **49.34 HOTA** (YOLO26x + ReID réglé), contre 42.72 pour
  le Kalman du sujet sur Yolov5l.

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
- `Appearance.py` : descripteurs d'apparence (histogramme de couleurs, MobileNetV3).
- `tune_appearance.py` : réglage et évaluation du tracker avec apparence (Partie IV).

Le filtre est implémenté localement avec NumPy, conformément à la partie II
du sujet (§ 1), puis importé par le tracker comme demandé au § 2.1.
Son constructeur reçoit les six paramètres `dt`, `u_x`, `u_y`, `std_acc`,
`x_std_meas`, `y_std_meas`. Chaque piste initialise ensuite son propre état
`[cx, cy, 0, 0]` à partir du centroïde de sa détection. Les six paramètres
sont explicités en tête de `main.py` ; leurs valeurs numériques
(hors accélération et vitesse initiale nulles) sont des choix du projet,
car le sujet n’en impose pas.
