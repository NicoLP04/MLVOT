# HANDOFF — MLVOT project (state of work, for another Claude instance)

Written 2026-11-20. Everything below was verified in the session unless marked
**(unverified)**. Project dir: `/home/nico/Documents/EPITA/ING3/MLVOT/MLVOT`.

## 1. What the project is

EPITA ING3 "MLVOT" course project: multi-object tracking (MOT) on one video
sequence, **ADL-Rundle-6** (525 frames, 1920x1080, 30 fps, 24 gt identities,
5009 gt boxes). Done in pairs, graded on the project (70 %) and a 7-minute oral
(30 %). Subject: `TP_MLVOT_2026.pdf`; course: `Support_cours.pdf`.

Subject parts:
- **Part I (15 pts)**: IoU tracker + Hungarian algorithm, track management,
  display, video, MOT-format result file (named like the sequence).
- **Part II (15 pts)**: add a Kalman filter per track (`KalmanFilter.py`, class
  `KalmanFilter` with `__init__`, `predict`, `update`), same association.
- **Part III (10 pts)**: evaluate with TrackEval: HOTA, IDF1, IDSW, FPS; comparison
  table; analysis (fast motion, disappearance, overlap; does Kalman help; where
  insufficient); representative visualizations.
- **Part IV (30 pts)**: improve the Kalman tracker (ReID/appearance, better
  detector, confidence-aware association, occlusion handling...), evaluate against
  Part II with the same protocol, report HOTA/IDF1/IDSW/FPS, strengths/limits,
  visualizations.

## 2. User preferences (follow them)

- The user writes **French**; answer in French.
- Code must be **simple and readable**: explicit loops over clever one-liners,
  **typed functions**, minimal files. User explicitly asked for **only two code
  files for Parts I-II** (`main.py` + `KalmanFilter.py`) and to follow the subject's
  names exactly (`KalmanFilter.py`, result file `ADL-Rundle-6.txt`).
- The user wants to **understand everything** (asked for line-by-line explanations,
  matrix-by-matrix explanations). Explain, do not just deliver.
- Be honest about uncertainty; the user pastes results back and relies on them.
- A "ponytail" lazy-developer skill was active early on (shortest working code, no
  speculative abstractions). The user's later requests (readability, types,
  naive-then-complex comparison) take precedence.
- uv is the package manager (`uv run ...`). The user commits/pushes themselves;
  **do not commit unless asked**.

## 3. Repository layout (current)

| File | Role |
|---|---|
| `main.py` | The tracker: detection loading, `iou`, `match` (Hungarian), `Track`, `KalmanTrack`, `AppearanceTrack`, `run_tracker`, drawing, CLI, export |
| `KalmanFilter.py` | Kalman filter, 6-parameter `__init__` exactly as the subject; the track sets the initial state `kf.x[:, 0] = [cx, cy, 0, 0]` |
| `Appearance.py` | Part IV descriptors (`color_histogram`, `CnnDescriptor` = MobileNetV3-small ImageNet), `get_features` with disk cache |
| `evaluate.py` | Part III: runs IoU/Kalman on the 3 detectors, times them, calls TrackEval, prints/saves table |
| `tune.py` | Hyperparameter grid for IoU/Kalman (Part III extra) |
| `tune_appearance.py` | Part IV grid + final comparison; `uv run tune_appearance.py hist\|cnn` |
| `README.md` | Install/usage + all results (kept up to date) |
| `figures/` | 5 PNG (IoU top row vs Kalman bottom row, around identity switches) |
| `TrackEval/` | Clone of https://github.com/JonathonLuiten/TrackEval (gitignored; own `.git`) |
| `results/` | Generated, gitignored: `results.md`, `tuning.md`, `appearance_*.md`, `trackers/`, `features/` (37 MB) |
| `slides/` | User's own content, not touched by me |
| `HANDOFF.md` | This file |

Data: `ADL-Rundle-6/{img1,det,gt,seqinfo.ini}`; detections `Yolov5s`, `Yolov5l`,
`public-dataset` (`det/<name>/det.txt`; the YOLO files are space-separated,
public-dataset is comma-separated; `load_detections` handles both).

Git state at the end of the session: last user commit `df3a055 Modified Readme to
contain results`. **Uncommitted**: `README.md`, `evaluate.py`, `main.py`,
`pyproject.toml`, `uv.lock`, new `Appearance.py`, `tune_appearance.py`,
`HANDOFF.md`. To push for Part IV: those files (not `TrackEval/`, `results/`).

## 4. How to run

```bash
uv sync --locked
uv run main.py                                   # Part I (IoU), Yolov5l, writes ADL-Rundle-6.txt
uv run main.py --tracker kalman --out kalman.txt # Part II
uv run main.py --tracker appearance --descriptor hist|cnn --out app.txt   # Part IV
uv run main.py Yolov5s --tracker kalman --show --video out.mp4            # window + video
uv run evaluate.py          # needs TrackEval cloned at repo root; ~1 min
uv run tune.py              # ~2+ min
uv run tune_appearance.py hist   # then cnn (CNN feature extraction ~1-2 min per detector)
```

Always pass `--out` when comparing trackers: the default `ADL-Rundle-6.txt` is the
subject-mandated name and gets overwritten. `--show` printed harmless Qt/Wayland
warnings on the user's machine (`QT_QPA_PLATFORM=xcb` hides the first one).

## 5. Algorithm summary

**Part I** (`main.py`): per frame, build the IoU similarity matrix between track boxes
and detections (double loop calling `iou`), `linear_sum_assignment(maximize=True)`,
reject pairs below `IOU_THRESHOLD=0.3`; matched -> update box, `missed=0`;
unmatched tracks -> `missed += 1`; unmatched detections -> new `Track` (unique id);
tracks removed when `missed > MAX_MISSED_FRAMES=5`. Only tracks seen in the current
frame are written/drawn (no interpolation). Output rows
`frame,id,left,top,w,h,1,-1,-1,-1`.

**Part II**: `KalmanTrack(Track)`: `box` = Kalman-predicted box (predicted centroid +
last known width/height). Each frame: `predict()` on all tracks, `match()` on predicted
boxes, `update()` with the detection centroid for matched tracks, no correction for
unmatched ones. Defaults: `DT=1`, `U=0`, `STD_ACC=1`, `X/Y_STD_MEAS=5` (values are our
choice; the subject imposes none). Matrices A, B, H, Q, R, P0=I exactly as in the
subject (checked against the PDF images). Written box = the detection box.
Display: observed box coloured by id, predicted box in thin red.

**Part IV** (`AppearanceTrack(KalmanTrack)`): similarity
`S = APP_ALPHA*IoU + APP_BETA*clip(cos(appearance),0,1)` (subject's formula), threshold
`APP_THRESHOLD=0.4` on S, lost tracks kept `APP_MAX_MISSED=30` frames, track descriptor
= EMA (`APP_EMA=0.8`) of matched detection descriptors, Kalman noise
`APP_STD_ACC=0.5`, `APP_STD_MEAS=10` (applied in `main()` for this mode only, via
`global`). Defaults alpha=beta=0.5. Descriptors: `hist` = HSV (16 hue x 4 sat bins) of
top half and bottom half of the box, L2-normalised; `cnn` = torchvision
`mobilenet_v3_small` ImageNet weights, 128x256 crops, 576-d, L2-normalised (NOT
ReID-trained). Parts I/II code paths are unchanged (outputs verified byte-identical
before/after Part IV).

## 6. Results (ADL-Rundle-6, TrackEval, MOT15 benchmark mode)

### Part III (subject's default parameters)

| Detections | Tracker | HOTA | DetA | AssA | IDF1 | IDSW | FPS (tracking only) |
|---|---|---|---|---|---|---|---|
| Yolov5s | IoU | 39.83 | 44.87 | 36.28 | 49.58 | 64 | ~12000 |
| Yolov5s | Kalman | 41.26 | 44.91 | 38.83 | 51.70 | 63 | ~2200 |
| Yolov5l | IoU | 41.69 | 46.65 | 38.09 | 51.57 | 59 | ~8800 |
| Yolov5l | Kalman | 42.72 | 47.08 | 39.56 | 52.91 | 52 | ~1760 |
| public-dataset | IoU | 19.15 | 31.45 | 11.92 | 22.72 | 203 | ~5200 |
| public-dataset | Kalman | 20.94 | 31.33 | 14.42 | 25.06 | 183 | ~1280 |

FPS varies about 10 % between runs (best of 3 timing runs, tracking only: no image
reading, no drawing). Kalman is ~5x slower, still far above real time.

### Hyperparameter tuning (`tune.py`, tuned on Yolov5l, checked on the others)

- IoU alone: no real gain (best `iou>=0.5` HOTA 41.81 vs 41.69 default; worse on the
  other detections) -> defaults kept.
- Kalman: best `iou>=0.2, max_missed=30, std_acc=0.5, std_meas=10`:
  Yolov5l 42.72 -> 44.62 HOTA (IDF1 52.91 -> 57.67, IDSW 52 -> 41);
  Yolov5s 41.26 -> 43.97; public-dataset 20.94 -> 23.71. Gain holds on held-out
  detections. Optimum values sit at grid edges (`max_missed=30` max, `std_acc=0.5`
  min, `std_meas=10` max) so a wider grid could find more. These tuned values are
  **not** the defaults in `main.py` for `--tracker kalman` (only the appearance mode
  uses `STD_ACC/STD_MEAS` 0.5/10 and memory 30).

### Part IV (tuned on Yolov5l: 40 configs over beta, S-threshold, memory; best was
beta=0.5, threshold 0.4, memory 30 for both descriptors)

| Detections | Kalman (subject) | Kalman tuned | Appearance hist | Appearance CNN |
|---|---|---|---|---|
| Yolov5l HOTA (IDSW) | 42.72 (52) | 44.62 (41) | 45.14 (37) | **46.05 (28)** |
| Yolov5s HOTA (IDSW) | 41.26 (63) | 43.97 (61) | **44.36 (39)** | 42.05 (37) |
| public-dataset HOTA (IDSW) | 20.94 (183) | 23.71 (183) | **24.21 (161)** | 22.81 (181) |

(Full tables incl. DetA/AssA/IDF1 are in `README.md` and `results/appearance_*.md`.)

## 7. Findings / analysis (for the report and oral)

1. **Detector is the main ceiling**: with Yolov5l 28.7 % of gt boxes have no correct
   detection (Yolov5s 36 %, public-dataset 45 %). Trackers cannot change this (the
   written box is the detection). Kalman changes only AssA, DetA is unchanged.
2. **AssA < DetA everywhere** (YOLO: ~36-39 vs ~45-47; public-dataset 12-14 vs 31),
   so association is the weak point -> motivated Part IV's track-recovery direction.
3. **Long disappearances**: ~3/4 of identity switches (Yolov5l: 54/72 IoU, 52/66 Kalman
   in my own gt-matching count) occur after a gap > 5 frames, when the track is already
   deleted; median gap 13 frames, max 89. Clear example:
   `figures/B_both_switch_gap35_f141.png` (a small child hidden 35 frames behind two
   adults gets a new id in both trackers).
4. **Kalman helps modestly** with subject defaults (+1 to +1.8 HOTA); only 4 of 18
   short-gap switches avoided on Yolov5l, none on Yolov5s. Kalman profits from longer
   memory (tuning), IoU alone does not (`max_missed` 5->30 flat at 41.8).
5. **Fast motion**: no link with failures in this sequence (8.8 % of gt boxes are
   "fast" > 14 px/frame, vs 8 % of switches).
6. **Overlap/occlusion**: **inconclusive** — my "occluded" criterion flagged 71 % of all
   gt boxes (very crowded scene), so it does not discriminate. The `A_*` figures show
   overlapping people swapping ids but are cluttered; not cleanly interpreted.
7. **Part IV**: appearance's own contribution over *tuned* Kalman is small: +0.4-0.5
   HOTA with the histogram on all 3 detections. The CNN overfits the tuning set (best
   on Yolov5l, worse than tuned Kalman on the two held-out detections) — hypothesis
   **(unverified)**: ImageNet features are not discriminative between people; a
   ReID-trained model (OSNet) would probably do better (not tested; no trusted weight
   source found). Both lower IDSW on all detections. On Yolov5l the reduction is
   concentrated on gaps of 6-30 frames (Kalman tuned 30 -> hist 28 -> cnn 20 switches);
   gaps > 30 unchanged (11) because the memory is 30 frames.
8. Descriptor cost: whole `main.py` run without cache ~11 s (hist, ~45 img/s) vs ~74 s
   (CNN, ~7 img/s) on 12 CPU cores; no GPU on this machine.
9. Caveats: one sequence, one run, grid-tuned on the evaluated gt (hence held-out
   detections as a check — still the same sequence); sub-1-point HOTA differences are
   not necessarily significant; my gt-matching switch counts (72/66 for Yolov5l) differ
   from TrackEval's IDSW (59/52) but go the same direction.

## 8. Gotchas for whoever continues

- **Run commands from the repo root** (`cd /home/nico/Documents/EPITA/ING3/MLVOT/MLVOT`);
  the shell cwd once drifted into `TrackEval/` and `uv run` created a stray `.venv` there.
- **TrackEval + NumPy 2**: TrackEval uses removed aliases (`np.float`, ...). `evaluate.py`
  restores only the *missing* ones (`name not in vars(np)`). Overwriting `np.bool` breaks
  `numpy.ma` (this was a real bug). TrackEval prints a harmless
  `Error importing BURST ... pycocotools`.
- **TrackEval setup**: `BENCHMARK="MOT15"` (gt has class column -1), `SKIP_SPLIT_FOL=True`,
  `GT_FOLDER=<repo root>` (so `<root>/ADL-Rundle-6/gt/gt.txt`), seqmap written by
  `evaluate.py`; metrics HOTA, CLEAR, Identity; HOTA/DetA/AssA are means over the
  19 alpha thresholds, x100.
- **Track objects are mutated across frames**: `run_tracker` yields live `Track`
  objects. Never store `list(run_tracker(...))` and format later (this produced
  nonsense metrics, HOTA ~8, once). Format rows while iterating.
- `Track.next_id` is a class counter, never reset by `main.py`; `evaluate.py` resets it
  before each run.
- `tune*.py` mutate module globals of `main` (`IOU_THRESHOLD`, `MAX_MISSED_FRAMES`,
  `STD_ACC`, `X/Y_STD_MEAS`, `APP_*`); `match`/`KalmanTrack` read them at call time, so
  this works. `main()` for `--tracker appearance` overrides `STD_*` via `global`.
- **Permissions**: the Claude Code classifier blocked me once from running
  `evaluate.py` (it executes the downloaded TrackEval code); the user then ran it with
  `! uv run evaluate.py`. Later runs (`tune_appearance.py`, background) went through.
  If blocked, ask the user to run it and paste the output.
- **Dependencies added in Part IV**: `torch`, `torchvision` (CPU wheels via the
  `pytorch-cpu` index in `pyproject.toml`); NumPy went 2.5.5 -> 2.4.6. MobileNetV3
  weights download from `download.pytorch.org` on first CNN use. The `README`
  originally mentioned `.env.example` which does not exist in the repo; ignore.
- A `git stash` was run by mistake once during the Part II/I merge; it was recovered with
  `git stash pop` (nothing lost).
- `results/features/*.npz` are caches: delete them if descriptors or detection files
  change.

## 9. What is left / open decisions

1. **Part IV visualizations**: DONE (2026-11-20). `figures/C_*` (appearance keeps the id
   after 16- and 28-frame gaps) and `figures/D_*` (35-frame gap > memory; 1-frame swap in
   overlap), described in README. Generated by scratchpad scripts `events.py` +
   `figures_app.py` (not in the repo).
1b. **Detector + interpolation**: DONE (2026-11-20, see README "meilleur détecteur et
   interpolation"). `detect.py` (ultralytics YOLO26x, conf 0.7) -> `det/yolo26x`
   (+ `det/yolo26x_raw`, conf >= 0.05). Best HOTA 46.77 (Kalman tuned on yolo26x).
   94 % of never-detected gt boxes are >50 % covered by another gt person (hidden):
   detector ceiling is the annotation, not the model. `--interpolate` tested: ~0 effect
   (gaps mostly belong to tracks on unannotated people). Scratch scripts: tune_conf.py,
   det_quality.py, miss_stats.py, eval_interp.py, interp_check.py.
1c. **ReID descriptor**: DONE (2026-11-20). `ReidDescriptor` in Appearance.py
   (yolo26m-reid.onnx, MSMT17, 448x448, onnxruntime). Best grid: b0.5 t0.3 m30 (t at grid
   edge). HOTA Yolov5l 47.11, Yolov5s 44.91, public 26.06 (best everywhere, generalizes);
   yolo26x 46.82 (~= Kalman tuned 46.77), yolo26x+interp 47.28 (best overall).
1d. **Grid on yolo26x + ReID**: DONE (2026-11-20, scratch grid_y26_reid.py, 216 configs).
   Best b0.25 t0.15 m60 std_acc0.25 std_meas10 -> 49.34 HOTA (best overall; interp lowers
   it to 48.79). Does NOT transfer to Yolov5l/s/public (worse than b0.5 t0.3 m30). Optimum
   still at grid edges. SET AS DEFAULTS in main.py (APP_*, descriptor reid) on user's request;
   `uv run main.py yolo26x --tracker appearance` reproduces 49.34.
2. **Wider tuning grid** (max_missed 40-60, std_acc 0.1-0.5, std_meas 10-20) since optima
   are at the edges; optionally apply tuned Kalman values with per-tracker presets in
   `main.py` (today `iou`/`kalman` modes use the subject's defaults on purpose).
3. **Stronger ReID** (OSNet) to replace the ImageNet MobileNet; or an improved detector
   (Part IV option 2: run a recent YOLO offline, save `det.txt` in MOT format).
4. **Multi-sequence evaluation** if more sequences are available (only ADL-Rundle-6 here).
5. Report + oral (7 min + 3 min questions): the subject expects a demo, the technical
   choices, and the limitations that motivated the improvement (sections 6-7 above).
6. Commit/push (user does it): files listed in section 3; keep `TrackEval/` and `results/`
   out of git.
