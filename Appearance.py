"""Part IV: appearance descriptors. Each detection (image patch) becomes a unit-length vector.

Two descriptors, same interface (list of patches -> array of shape (n, d)):
  - color_histogram : naive, HSV histograms of the top and bottom halves of the box
  - CnnDescriptor   : pretrained MobileNetV3 features (needs torch, see pyproject.toml)
  - ReidDescriptor  : person re-identification model YOLO26 ReID (needs onnxruntime)

Since vectors are unit-length and non-negative-ish, the dot product of two descriptors
is their cosine similarity.
"""
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(ROOT, "results", "features")


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Image patch inside a box (left, top, width, height), clipped to the image."""
    h, w = image.shape[:2]
    left, top, width, height = box
    x0, y0 = max(0, int(left)), max(0, int(top))
    x1, y1 = min(w, int(left + width)), min(h, int(top + height))
    return image[y0:y1, x0:x1]


def color_histogram(patches: list[np.ndarray]) -> np.ndarray:
    """Naive descriptor: hue/saturation histograms of the top half (shirt) and bottom half (trousers)."""
    descriptors = []
    for patch in patches:
        if patch.shape[0] < 2 or patch.shape[1] < 1:   # box outside the image
            descriptors.append(np.zeros(128))
            continue
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        middle = len(hsv) // 2
        halves = []
        for half in (hsv[:middle], hsv[middle:]):
            hist = cv2.calcHist([half], [0, 1], None, [16, 4], [0, 180, 0, 256]).flatten()
            halves.append(unit(hist))
        descriptors.append(unit(np.concatenate(halves)))
    return np.array(descriptors)


class CnnDescriptor:
    """Pretrained MobileNetV3-small (ImageNet) used as a feature extractor."""

    def __init__(self) -> None:
        import torch
        import torchvision
        self.torch = torch
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.model = torchvision.models.mobilenet_v3_small(weights=weights)
        self.model.classifier = torch.nn.Identity()   # keep the 576-d pooled features
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(self, patches: list[np.ndarray]) -> np.ndarray:
        batch = []
        for patch in patches:
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                patch = np.zeros((256, 128, 3), np.uint8)
            rgb = cv2.cvtColor(cv2.resize(patch, (128, 256)), cv2.COLOR_BGR2RGB)
            batch.append((rgb.astype(np.float32) / 255 - self.mean) / self.std)
        x = self.torch.from_numpy(np.array(batch)).permute(0, 3, 1, 2)
        with self.torch.no_grad():
            features = self.model(x).numpy()
        return np.array([unit(f) for f in features])


class ReidDescriptor:
    """Person re-identification model (Ultralytics YOLO26m ReID, trained on MSMT17 to tell people apart)."""

    def __init__(self, model: str = "yolo26m-reid.onnx") -> None:
        import ast
        import onnxruntime
        from ultralytics.utils.downloads import attempt_download_asset
        self.session = onnxruntime.InferenceSession(attempt_download_asset(model))
        self.input_name = self.session.get_inputs()[0].name
        # the input size is dynamic in the graph; the training size is in the metadata ("[448, 448]")
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.height, self.width = ast.literal_eval(metadata["imgsz"])

    def __call__(self, patches: list[np.ndarray]) -> np.ndarray:
        batch = []
        for patch in patches:
            if patch.shape[0] < 2 or patch.shape[1] < 2:
                patch = np.zeros((self.height, self.width, 3), np.uint8)
            # same preprocessing as Ultralytics: RGB, resized (not padded), values in [0, 1]
            rgb = cv2.cvtColor(cv2.resize(patch, (self.width, self.height)), cv2.COLOR_BGR2RGB)
            batch.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255)
        features = self.session.run(None, {self.input_name: np.array(batch)})[0]
        return np.array([unit(f) for f in features])


def get_features(detector: str, descriptor: str, detections: dict) -> dict[int, np.ndarray]:
    """{frame: array (n_detections, d)}, in the same order as `detections`. Cached on disk."""
    path = os.path.join(FEATURES_DIR, f"{descriptor}_{detector}.npz")
    if os.path.exists(path):
        data = np.load(path)
        return {int(k): data[k] for k in data.files}

    if descriptor == "hist":
        describe = color_histogram
    elif descriptor == "cnn":
        describe = CnnDescriptor()
    else:
        describe = ReidDescriptor()
    features = {}
    for frame, boxes in detections.items():
        image = cv2.imread(os.path.join(ROOT, "ADL-Rundle-6", "img1", f"{frame:06d}.jpg"))
        features[frame] = describe([crop(image, box) for box in boxes])

    os.makedirs(FEATURES_DIR, exist_ok=True)
    np.savez(path, **{str(f): v for f, v in features.items()})
    return features
