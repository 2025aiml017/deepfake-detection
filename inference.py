"""Video -> REAL / FAKE: frame sampling, RetinaFace detection, eye alignment and ConvNeXt-Tiny scoring.

Mirrors the training notebook (sections 3, 9 and 10) so the classifier sees faces cropped exactly as in training.
The face detector is insightface's RetinaFace (det_10g.onnx from the buffalo_l pack) run directly with ONNX
Runtime; the insightface package itself is not needed.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "deepfake_convnext_tiny.pt"
DETECTOR_PATH = ROOT / "models" / "det_10g.onnx"
DETECTOR_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DETECTOR_SHA256 = "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"

# Same values as the notebook's Config and alignment cell.
EDGE_TRIM = 0.05
DET_SIZE = 640
NMS_THRESHOLD = 0.4
EYE_MARGIN_X = 0.35
EYE_LINE_Y = 0.35

# Hosted containers often report the host's core count; more threads than cores only adds contention.
NUM_THREADS = max(1, min(4, os.cpu_count() or 1))


def ensure_detector(path: Path = DETECTOR_PATH) -> Path:
    """Download det_10g.onnx from insightface's official release on first use (about 290 MB, done once)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        archive, staged = Path(tmp) / "buffalo_l.zip", Path(tmp) / path.name
        with urllib.request.urlopen(DETECTOR_URL, timeout=60) as response, open(archive, "wb") as out:
            shutil.copyfileobj(response, out, length=1 << 20)
        with zipfile.ZipFile(archive) as bundle, bundle.open("det_10g.onnx") as src, open(staged, "wb") as dst:
            shutil.copyfileobj(src, dst)
        digest = hashlib.sha256(staged.read_bytes()).hexdigest()
        if digest != DETECTOR_SHA256:
            raise RuntimeError(f"Unexpected det_10g.onnx checksum {digest}")
        staged.replace(path)
    return path


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return np.stack([points[:, 0] - distance[:, 0], points[:, 1] - distance[:, 1],
                     points[:, 0] + distance[:, 2], points[:, 1] + distance[:, 3]], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    coords = []
    for i in range(0, distance.shape[1], 2):
        coords += [points[:, i % 2] + distance[:, i], points[:, i % 2 + 1] + distance[:, i + 1]]
    return np.stack(coords, axis=-1)


def nms(dets: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        overlap = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


class FaceDetector:
    """insightface's RetinaFace.detect() for det_10g.onnx, as called by FaceAnalysis(det_size=(640, 640))."""

    strides = (8, 16, 32)
    anchors_per_location = 2

    def __init__(self, model_path: Path, score_threshold: float = 0.5):
        options = ort.SessionOptions()
        options.intra_op_num_threads = NUM_THREADS
        self.session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise ValueError(f"Expected det_10g.onnx (9 outputs), got {len(self.output_names)} outputs")
        self.score_threshold = score_threshold
        self._centers: dict[int, np.ndarray] = {}

    def _anchor_centers(self, stride: int) -> np.ndarray:
        if stride not in self._centers:
            size = DET_SIZE // stride
            centers = np.stack(np.mgrid[:size, :size][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape(-1, 2)
            self._centers[stride] = np.stack([centers] * self.anchors_per_location, axis=1).reshape(-1, 2)
        return self._centers[stride]

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return boxes [N, 5] (x1, y1, x2, y2, score) and landmarks [N, 5, 2] for a BGR image."""
        ratio = image.shape[0] / image.shape[1]
        if ratio > 1:
            new_height, new_width = DET_SIZE, int(DET_SIZE / ratio)
        else:
            new_height, new_width = int(DET_SIZE * ratio), DET_SIZE
        scale = new_height / image.shape[0]
        canvas = np.zeros((DET_SIZE, DET_SIZE, 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = cv2.resize(image, (new_width, new_height))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (DET_SIZE, DET_SIZE), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        if outputs[0].ndim == 3:  # some exports keep the batch dimension
            outputs = [output[0] for output in outputs]

        scores, boxes, landmarks = [], [], []
        for level, stride in enumerate(self.strides):
            centers = self._anchor_centers(stride)
            keep = np.where(outputs[level] >= self.score_threshold)[0]
            scores.append(outputs[level][keep])
            boxes.append(distance2bbox(centers, outputs[level + 3] * stride)[keep])
            landmarks.append(distance2kps(centers, outputs[level + 6] * stride).reshape(len(centers), -1, 2)[keep])

        scores = np.vstack(scores)
        order = scores.ravel().argsort()[::-1]
        dets = np.hstack((np.vstack(boxes) / scale, scores)).astype(np.float32, copy=False)[order]
        keep = nms(dets, NMS_THRESHOLD)
        return dets[keep], (np.vstack(landmarks) / scale)[order][keep]


def largest_face(boxes: np.ndarray) -> int | None:
    if len(boxes) == 0:
        return None
    return int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray | None:
    left_eye, right_eye = landmarks[0], landmarks[1]
    dx, dy = (right_eye - left_eye).tolist()
    distance = math.hypot(dx, dy)
    if distance < 1e-3:
        return None
    center = tuple(((left_eye + right_eye) / 2).tolist())
    scale = (1 - 2 * EYE_MARGIN_X) * size / distance
    matrix = cv2.getRotationMatrix2D(center, math.degrees(math.atan2(dy, dx)), scale)
    matrix[0, 2] += size * 0.5 - center[0]
    matrix[1, 2] += size * EYE_LINE_Y - center[1]
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def to_nchw(tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Feature map of a Grad-CAM layer as [N, C, H, W], whatever layout the model uses."""
    if tensor.ndim == 3:
        tokens = tensor[:, model.num_prefix_tokens:]
        side = math.isqrt(tokens.shape[1])
        return tokens.transpose(1, 2).reshape(len(tensor), -1, side, side)
    if getattr(model, "output_fmt", "NCHW") == "NHWC":
        return tensor.permute(0, 3, 1, 2)
    return tensor


def heatmap_overlay(face: np.ndarray, heatmap: np.ndarray, color: tuple[int, int, int] = (235, 104, 52)) -> np.ndarray:
    """Show a heatmap on an RGB face: dim the face where the heatmap is low and tint it towards `color` where it is
    high. A tint alone barely shows on skin tones; the dimming makes the hot areas stand out on any face."""
    heat = heatmap[..., None]
    lit = face * (0.35 + 0.65 * heat)
    return (lit * (1 - 0.45 * heat) + np.array(color) * 0.45 * heat).astype(np.uint8)


class DeepfakeClassifier:
    """The exported inference bundle (timm model, preprocessing settings and the video-level threshold) and the
    reference file beside it (Grad-CAM layer, calibration and inconclusive range; see scripts/make_reference.py)."""

    def __init__(self, bundle_path: Path = MODEL_PATH):
        torch.set_num_threads(NUM_THREADS)
        bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
        self.model = timm.create_model(bundle["timm_name"], pretrained=False, num_classes=1, **bundle["model_kwargs"])
        self.model.load_state_dict({name: tensor.float() for name, tensor in bundle["state_dict"].items()})
        self.model.eval().requires_grad_(False)
        self.name = bundle["model_key"]
        self.threshold = float(bundle["threshold"])
        self.align_size = int(bundle["align_size"])
        self.min_face_score = float(bundle["min_face_score"])
        self.transform = transforms.Compose([
            transforms.Resize((bundle["img_size"], bundle["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(bundle["mean"], bundle["std"]),
        ])

        reference_path = bundle_path.with_suffix(".json")
        self.reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
        if self.reference and not math.isclose(self.reference["threshold"], self.threshold, abs_tol=1e-6):
            raise ValueError(f"{reference_path.name} was built for a different model file; rerun make_reference.py")
        self.calibration = self.reference.get("calibration")
        self.inconclusive_range = self.reference.get("inconclusive")

    def _batch(self, faces: list[np.ndarray]) -> torch.Tensor:
        return torch.stack([self.transform(Image.fromarray(face)) for face in faces])

    @torch.inference_mode()
    def predict(self, crops: list[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """P(fake) for each BGR face crop."""
        probs = []
        for start in range(0, len(crops), batch_size):
            batch = self._batch([cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops[start:start + batch_size]])
            probs.append(torch.sigmoid(self.model(batch)).squeeze(1).numpy())
        return np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)

    def grad_cam(self, faces: list[np.ndarray], batch_size: int = 8) -> list[np.ndarray]:
        """Grad-CAM of the FAKE logit for each RGB face (notebook section 9): a heatmap of the face's size, scaled
        to 0-1 per face, that is high where the face pushed the score towards FAKE."""
        layer = self.model.get_submodule(self.reference["cam_layer"])
        captured, owner = {}, threading.get_ident()

        def start_graph_here(_module, _inputs, output):
            # The app shares one model between sessions, each in its own thread: leave their forward passes alone.
            if threading.get_ident() != owner:
                return None
            # The layers before this one run without autograd; gradients are only needed from here on.
            captured["activation"] = output.detach().requires_grad_(True)
            return captured["activation"]

        heatmaps = []
        handle = layer.register_forward_hook(start_graph_here)
        try:
            for start in range(0, len(faces), batch_size):
                chunk = faces[start:start + batch_size]
                with torch.enable_grad():
                    logits = self.model(self._batch(chunk))
                    gradient, = torch.autograd.grad(logits.sum(), captured["activation"])
                activation = to_nchw(captured["activation"].detach(), self.model)
                gradient = to_nchw(gradient, self.model)
                cams = (gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1).relu().numpy()
                for cam, face in zip(cams, chunk):
                    cam = cv2.resize(cam, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_LINEAR)
                    heatmaps.append((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))
        finally:
            handle.remove()
        return heatmaps

    def calibrated(self, score: float) -> float | None:
        """P(fake) of a video score after Platt scaling on validation videos, taking real and fake videos as equally
        likely. None when the reference file has no calibration."""
        if self.calibration is None or math.isnan(score):
            return None
        clip = self.calibration["logit_clip"]
        score = min(max(score, clip), 1 - clip)
        z = self.calibration["slope"] * math.log(score / (1 - score)) + self.calibration["intercept"]
        return 1 / (1 + math.exp(-z))

    def is_inconclusive(self, score: float) -> bool:
        """True when the score is in the range where both real and fake validation videos scored."""
        return self.inconclusive_range is not None and self.inconclusive_range[0] <= score <= self.inconclusive_range[1]


def frame_indices(n_total: int, count: int) -> np.ndarray:
    if n_total <= 0:
        return np.array([], dtype=int)
    first, last = int(n_total * EDGE_TRIM), int(n_total * (1 - EDGE_TRIM)) - 1
    if last <= first:
        first, last = 0, n_total - 1
    return np.linspace(first, last, min(count, last - first + 1)).astype(int)


def count_frames(path: Path) -> int:
    capture, total = cv2.VideoCapture(str(path)), 0
    while capture.grab():
        total += 1
    capture.release()
    return total


def sample_frames(path: Path, count: int) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (index, BGR frame) for `count` evenly spaced frames, skipping the first and last 5% like training."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:  # some containers (e.g. WebM) do not store a frame count
        total = count_frames(path)
    missed = []
    for index in frame_indices(total, count).tolist():
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            yield index, frame
        else:
            missed.append(index)
    capture.release()

    if missed:  # seeking failed for these frames: decode sequentially instead
        capture, index, wanted = cv2.VideoCapture(str(path)), 0, set(missed)
        while wanted and capture.grab():
            if index in wanted:
                ok, frame = capture.retrieve()
                if ok:
                    yield index, frame
                wanted.discard(index)
            index += 1
        capture.release()


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int

    @property
    def duration(self) -> float | None:
        return self.frames / self.fps if self.fps > 0 and self.frames > 0 else None


def video_info(path: Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    info = VideoInfo(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     float(capture.get(cv2.CAP_PROP_FPS) or 0.0), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.release()
    return info


@dataclass
class VideoResult:
    verdict: str  # "REAL", "FAKE" or "NO FACE"
    score: float  # mean P(fake) over the detected faces; nan when no face was found
    threshold: float
    frames_read: int
    frame_indices: list[int]  # frame of each detected face
    probs: list[float]  # P(fake) of each detected face
    crops: list[np.ndarray]  # aligned faces as RGB, for display
    info: VideoInfo
    seconds: float
    calibrated: float | None = None  # P(fake) after calibration; None without a face or a reference file
    inconclusive: bool = False  # score in the range where real and fake validation videos overlap
    no_face_frames: list[int] = field(default_factory=list)  # sampled frames where no face was found


def analyse_video(path: Path, detector: FaceDetector, classifier: DeepfakeClassifier, num_frames: int = 32,
                  progress: Callable[[float, str], None] | None = None) -> VideoResult:
    """Score a video: mean P(fake) over the largest face in `num_frames` evenly spaced frames."""
    start = time.perf_counter()
    report = progress or (lambda fraction, message: None)
    info = video_info(path)
    crops, indices, no_face, frames_read = [], [], [], 0
    for index, frame in sample_frames(path, num_frames):
        frames_read += 1
        report(0.9 * min(frames_read / num_frames, 1.0), f"Finding faces: frame {frames_read} of {num_frames}")
        boxes, landmarks = detector.detect(frame)
        face = largest_face(boxes)
        crop = None if face is None else align_face(frame, landmarks[face], classifier.align_size)
        if crop is not None:
            crops.append(crop)
            indices.append(index)
        else:
            no_face.append(index)
    if frames_read == 0:
        raise ValueError("No frames could be read from this video.")

    report(0.9, f"Scoring {len(crops)} faces")
    probs = classifier.predict(crops)
    score = float(probs.mean()) if len(probs) else float("nan")
    verdict = "NO FACE" if not len(probs) else ("FAKE" if score >= classifier.threshold else "REAL")
    return VideoResult(verdict, score, classifier.threshold, frames_read, indices, probs.tolist(),
                       [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in crops], info,
                       time.perf_counter() - start, calibrated=classifier.calibrated(score),
                       inconclusive=classifier.is_inconclusive(score), no_face_frames=sorted(no_face))
