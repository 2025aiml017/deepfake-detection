# %% [markdown]
# # Low resolution and speed tests
#
# Two experiments for the final presentation, run with the deployed Streamlit app's own pipeline
# (`inference.py` and `models/deepfake_convnext_tiny.pt` from the project's GitHub repository):
#
# 1. **Low resolution:** holdout videos (people never seen in training) are downscaled to 360p, 240p, 144p
#    and 96p before face detection. Are the fake videos still caught, and do the real ones stay REAL?
# 2. **Speed:** time to score one video, by video size, number of frames and hardware.
#
# It also records the resolution and length of every Celeb-DF v2 video, for the data-distribution slides.
#
# **On Kaggle:** add the *Celeb DF (v2)* input, set *Accelerator* to *GPU T4* and *Internet* to *On*, then
# *Save Version → Save & Run All*. When the version finishes (about an hour), download
# `lowres_speed_results.zip` from its *Output* tab.
#
# Running the file locally (`python experiments/lowres_speed.py`) does the CPU speed test and a small low-resolution
# check on the two demo videos in `~/Downloads`.

# %%
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ON_KAGGLE = Path("/kaggle/input").exists()
COMMIT = "742e0718f84c6d518f3f84f0cd30cec79398322e"  # repository version deployed to Streamlit
REPO_RAW = f"https://raw.githubusercontent.com/2025aiml017/deepfake-detection/{COMMIT}"


def pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args], check=True)


if ON_KAGGLE:
    APP_DIR, OUT = Path("/kaggle/working/app"), Path("/kaggle/working/lowres_speed_results")
    pip("timm==1.0.11", "--no-deps")  # the version that trained the checkpoint
    # GPU build of ONNX Runtime so the face detector can also run on the T4 (falls back to CPU if it fails).
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime"], check=False)
    try:
        pip("onnxruntime-gpu==1.30.0")
    except subprocess.CalledProcessError:
        pip("onnxruntime==1.30.0")
    (APP_DIR / "models").mkdir(parents=True, exist_ok=True)
    for name in ("inference.py", "models/deepfake_convnext_tiny.pt"):
        if not (APP_DIR / name).exists():
            urllib.request.urlretrieve(f"{REPO_RAW}/{name}", APP_DIR / name)
else:
    APP_DIR = Path.cwd() if (Path.cwd() / "inference.py").exists() else Path(__file__).resolve().parents[1]
    OUT = Path(os.environ.get("LOWRES_SPEED_OUT", APP_DIR / "experiments" / "results_local"))
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(APP_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import inference  # noqa: E402

try:
    ort.preload_dlls()  # CUDA and cuDNN libraries that ship with PyTorch
except Exception as error:  # older ONNX Runtime or no GPU libraries: the detector stays on CPU
    print("preload_dlls:", error)
CUDA_DETECTOR = "CUDAExecutionProvider" in ort.get_available_providers()
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def cpu_name() -> str:
    try:
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


HARDWARE = {"gpu": GPU, "cpu": cpu_name(), "cpu_threads_used": inference.NUM_THREADS,
            "onnxruntime_providers": ort.get_available_providers(), "torch": torch.__version__}
print(json.dumps(HARDWARE, indent=2))

# %% [markdown]
# ## Pipeline on a chosen device
# The app runs on CPU. These subclasses only change *where* the same code runs: the ONNX Runtime provider of the
# RetinaFace detector and the PyTorch device of the ConvNeXt-Tiny classifier.

# %%
class Detector(inference.FaceDetector):
    def __init__(self, model_path: Path, score_threshold: float, on_gpu: bool):
        super().__init__(model_path, score_threshold)
        if on_gpu:
            try:
                self.session = ort.InferenceSession(str(model_path),
                                                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as error:
                print("CUDA face detector unavailable, using CPU:", error)
        self.device = "GPU" if self.session.get_providers()[0] == "CUDAExecutionProvider" else "CPU"


class Classifier(inference.DeepfakeClassifier):
    def __init__(self, bundle_path: Path, device: str):
        super().__init__(bundle_path)
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.inference_mode()
    def predict(self, crops: list, batch_size: int = 64) -> np.ndarray:
        probs = []
        for start in range(0, len(crops), batch_size):
            batch = torch.stack([self.transform(inference.Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                                 for c in crops[start:start + batch_size]]).to(self.device)
            probs.append(torch.sigmoid(self.model(batch)).squeeze(1).float().cpu().numpy())
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return np.concatenate(probs) if probs else np.empty(0, dtype=np.float32)


DETECTOR_FILE = inference.ensure_detector(APP_DIR / "models" / "det_10g.onnx")
classifier = Classifier(APP_DIR / "models" / "deepfake_convnext_tiny.pt", "cuda" if GPU else "cpu")
detector = Detector(DETECTOR_FILE, classifier.min_face_score, on_gpu=CUDA_DETECTOR)
THRESHOLDS = {"video": classifier.threshold, "frame": float(torch.load(
    APP_DIR / "models" / "deepfake_convnext_tiny.pt", map_location="cpu", weights_only=True)["frame_threshold"])}
print(f"classifier on {classifier.device} | face detector on {detector.device} | thresholds {THRESHOLDS}")


def face_crops(frames: list, det: inference.FaceDetector, align_size: int) -> list:
    crops = []
    for frame in frames:
        boxes, landmarks = det.detect(frame)
        face = inference.largest_face(boxes)
        crop = None if face is None else inference.align_face(frame, landmarks[face], align_size)
        if crop is not None:
            crops.append(crop)
    return crops

# %% [markdown]
# ## Videos and holdout split
# The split is rebuilt with the training notebook's own code (section 2) and checked against the 809 holdout videos
# of the run that produced the checkpoint.

# %%
HOLDOUT_SHA256 = "a4cd898e47ed692b9d9444c161ee80ba51e4d0994262c41b930c45b5682cda74"
REAL_DIRS, FAKE_DIRS, TEST_LIST = ("Celeb-real", "YouTube-real"), ("Celeb-synthesis",), "List_of_testing_videos.txt"
SPLIT_TARGETS = {"train": 1 - 0.15 - 0.10, "val": 0.15, "holdout": 0.10}  # same expression as the notebook's Config


def find_celeb_df_root() -> Path:
    for p in [Path("/kaggle/input"), *Path("/kaggle/input").rglob("*")]:
        if p.is_dir() and all((p / d).is_dir() for d in REAL_DIRS + FAKE_DIRS) and (p / TEST_LIST).exists():
            return p
    raise FileNotFoundError("Attach the Celeb DF (v2) dataset under Input.")


def parse_identities(stem: str):
    import re
    if m := re.match(r"^id(\d+)_id(\d+)_\d+$", stem):
        return f"id{m[1]}", f"id{m[2]}"
    if m := re.match(r"^id(\d+)_\d+$", stem):
        return f"id{m[1]}", None
    if m := re.match(r"^(\d+)$", stem):
        return f"yt_{m[1]}", None
    raise ValueError(stem)


def build_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for folder in REAL_DIRS + FAKE_DIRS:
        for path in sorted((root / folder).glob("*.mp4")):
            id_a, id_b = parse_identities(path.stem)
            rows.append({"rel_path": f"{folder}/{path.name}", "folder": folder, "stem": path.stem,
                         "label": int(folder in FAKE_DIRS), "id_a": id_a, "id_b": id_b})
    videos = pd.DataFrame(rows)
    lines = [line.split() for line in (root / TEST_LIST).read_text().splitlines()]
    official = {path for flag, path in (row for row in lines if len(row) == 2)}
    videos["is_official_test"] = videos.rel_path.isin(official)

    parent = {}

    def find(identity):
        parent.setdefault(identity, identity)
        while parent[identity] != identity:
            parent[identity] = parent[parent[identity]]
            identity = parent[identity]
        return identity

    for id_a, id_b in zip(videos.id_a, videos.id_b):
        find(id_a)
        if isinstance(id_b, str):
            parent[find(id_b)] = find(id_a)
    videos["component"] = videos.id_a.map(find)

    def greedy_assign(sizes):
        total, filled, assignment = sizes.sum(), dict.fromkeys(SPLIT_TARGETS, 0), {}
        for component, size in sizes.items():
            split = max(SPLIT_TARGETS, key=lambda s: SPLIT_TARGETS[s] * total - filled[s])
            assignment[component] = split
            filled[split] += size
        return assignment

    stats = videos[~videos.is_official_test].groupby("component").label.agg(n_fake="sum", n_videos="size")
    stats["n_real"] = stats.n_videos - stats.n_fake
    with_fakes = stats[stats.n_fake > 0].n_fake.sort_values(ascending=False, kind="stable")
    real_only = stats[stats.n_fake == 0].n_real.sort_values(ascending=False, kind="stable")
    assignment = {**greedy_assign(with_fakes), **greedy_assign(real_only)}
    videos["split"] = videos.component.map(assignment).where(~videos.is_official_test, "test")
    return videos


if ON_KAGGLE:
    DATA_ROOT = find_celeb_df_root()
    VIDEOS = build_manifest(DATA_ROOT)
    holdout_stems = sorted(VIDEOS[VIDEOS.split == "holdout"].stem)
    assert hashlib.sha256("\n".join(holdout_stems).encode()).hexdigest() == HOLDOUT_SHA256, "holdout split differs"
    print(VIDEOS.groupby(["split", "label"]).size().unstack())
else:  # the two holdout demo videos saved by the notebook
    DATA_ROOT = Path.home() / "Downloads"
    VIDEOS = pd.DataFrame([
        {"rel_path": "id49_0000.mp4", "folder": "Celeb-real", "stem": "id49_0000", "label": 0, "split": "holdout"},
        {"rel_path": "id49_id50_0000.mp4", "folder": "Celeb-synthesis", "stem": "id49_id50_0000", "label": 1,
         "split": "holdout"},
    ])


def video_props(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, frames = float(capture.get(cv2.CAP_PROP_FPS) or 0), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return {"width": width, "height": height, "fps": fps, "frames": frames,
            "duration_s": frames / fps if fps > 0 else float("nan"), "size_mb": path.stat().st_size / 1e6}

# %% [markdown]
# ## 1. Dataset statistics (all 6,529 videos)

# %%
if ON_KAGGLE:
    stats = pd.DataFrame([{**row, **video_props(DATA_ROOT / row["rel_path"])}
                          for row in VIDEOS[["rel_path", "folder", "stem", "label", "split"]].to_dict("records")])
    stats.to_csv(OUT / "dataset_videos.csv", index=False)
    print(stats.groupby("folder")[["width", "height", "duration_s", "frames", "size_mb"]].describe().T.round(1))

# %% [markdown]
# ## 2. Low resolution: does downscaling hide the fake?
# Each holdout video is read once (evenly spaced frames, as in the app), then every frame is downscaled to the
# target height before face detection, alignment and scoring. Frames already at or below a height are not upscaled.

# %%
HEIGHTS = [0, 360, 240, 144, 96]  # 0 = original resolution
if ON_KAGGLE:
    FRAMES_PER_VIDEO = 16 if detector.device == "GPU" else 8  # keeps the CPU fallback near an hour
    holdout = VIDEOS[VIDEOS.split == "holdout"]
    fakes = holdout[holdout.label == 1]
    EXAMPLES = {"id49_id50_0000", "id49_0000", *fakes[fakes.stem != "id49_id50_0000"].sample(3, random_state=1).stem}
    if detector.device == "CPU":
        fakes = pd.concat([fakes[fakes.stem.isin(EXAMPLES)],
                           fakes[~fakes.stem.isin(EXAMPLES)].sample(n=196, random_state=0)])
    STUDY = pd.concat([holdout[holdout.label == 0], fakes])
else:
    FRAMES_PER_VIDEO, STUDY = 16, VIDEOS
    EXAMPLES = {"id49_id50_0000", "id49_0000"}
(OUT / "examples").mkdir(exist_ok=True)


def downscale(frame: np.ndarray, height: int) -> np.ndarray:
    if not height or frame.shape[0] <= height:
        return frame
    width = round(frame.shape[1] * height / frame.shape[0])
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


rows, start = [], time.time()
for n, video in enumerate(STUDY.itertuples(), start=1):
    path = DATA_ROOT / video.rel_path
    frames = [frame for _, frame in inference.sample_frames(path, FRAMES_PER_VIDEO)]
    for height in HEIGHTS:
        small = [downscale(frame, height) for frame in frames]
        crops = face_crops(small, detector, classifier.align_size)
        probs = classifier.predict(crops)
        rows.append({"stem": video.stem, "label": video.label, "folder": video.folder, "target_height": height,
                     "width": small[0].shape[1], "height": small[0].shape[0], "frames": len(small), "faces": len(crops),
                     "score": float(probs.mean()) if len(probs) else np.nan, "frame_scores": json.dumps(
                         [round(float(p), 5) for p in probs])})
        if video.stem in EXAMPLES and crops:
            tag = f"{video.stem}_{height or 'orig'}"
            cv2.imwrite(str(OUT / "examples" / f"{tag}_frame.jpg"), small[len(small) // 2], [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(OUT / "examples" / f"{tag}_face.png"), crops[len(crops) // 2])
    if n % 50 == 0 or n == len(STUDY):
        print(f"{n}/{len(STUDY)} videos, {time.time() - start:.0f} s")

lowres = pd.DataFrame(rows)
lowres.to_csv(OUT / "lowres_videos.csv", index=False)


def summarise(group: pd.DataFrame) -> pd.Series:
    scored = group.dropna(subset=["score"])
    out = {"videos": len(group), "face_found_pct": 100 * group.faces.sum() / group.frames.sum(),
           "videos_without_face": int(group.score.isna().sum()), "mean_score": scored.score.mean()}
    for name, t in THRESHOLDS.items():
        out[f"pct_called_fake_{name}_threshold"] = 100 * (scored.score >= t).mean() if len(scored) else np.nan
    return pd.Series(out)


def auc(labels: pd.Series, scores: pd.Series) -> float:
    """Probability that a fake video scores above a real one (ties count half); videos without a face are left out."""
    known = scores.notna()
    fake, real = scores[known & (labels == 1)].to_numpy(), scores[known & (labels == 0)].to_numpy()
    if not len(fake) or not len(real):
        return np.nan
    diff = fake[:, None] - real[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


summary = lowres.groupby(["target_height", "label"]).apply(summarise, include_groups=False).reset_index()
summary["video_auc"] = summary.target_height.map({h: auc(g.label, g.score) for h, g in lowres.groupby("target_height")})
summary.to_csv(OUT / "lowres_summary.csv", index=False)
print(summary.round(3).to_string(index=False))

# %% [markdown]
# ## 3. Speed: video size × number of frames × hardware
# One holdout video is re-encoded at several sizes (and once looped to about a minute). For each size, frame count
# and hardware setup the full pipeline runs once to warm up, then three timed runs; the median of each stage is kept.

# %%
BASE = DATA_ROOT / VIDEOS.set_index("stem").loc["id49_id50_0000", "rel_path"]
SPEED_DIR = OUT / "speed_videos"
SPEED_DIR.mkdir(exist_ok=True)


def make_variant(src: Path, dst: Path, size=None, loops: int = 1) -> Path:
    capture = cv2.VideoCapture(str(src))
    fps, frames = capture.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame if size is None else cv2.resize(
            frame, size, interpolation=cv2.INTER_AREA if size[0] < frame.shape[1] else cv2.INTER_CUBIC))
    capture.release()
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, frames[0].shape[1::-1])
    for _ in range(loops):
        for frame in frames:
            writer.write(frame)
    writer.release()
    return dst


SPEED_VIDEOS = {
    "640x360": make_variant(BASE, SPEED_DIR / "640x360.mp4", (640, 360)),
    "856x478 (original)": BASE,
    "1280x720": make_variant(BASE, SPEED_DIR / "1280x720.mp4", (1280, 720)),
    "1920x1080": make_variant(BASE, SPEED_DIR / "1920x1080.mp4", (1920, 1080)),
    "856x478, looped to ~1 min": make_variant(BASE, SPEED_DIR / "856x478_looped.mp4", loops=6),
}
FRAME_COUNTS = [8, 16, 32, 64]

if GPU:
    SETUPS = [("NVIDIA T4: model and face detector on GPU", "cuda", True)] if CUDA_DETECTOR else []
    SETUPS += [("NVIDIA T4: model on GPU, face detector on CPU", "cuda", False), ("Kaggle CPU only", "cpu", False)]
else:
    SETUPS = [(f"{HARDWARE['cpu']} CPU only", "cpu", False)]


def timed_run(path: Path, det: Detector, clf: Classifier, num_frames: int) -> dict:
    t0 = time.perf_counter()
    frames = [frame for _, frame in inference.sample_frames(path, num_frames)]
    t1 = time.perf_counter()
    crops = face_crops(frames, det, clf.align_size)
    t2 = time.perf_counter()
    probs = clf.predict(crops)
    t3 = time.perf_counter()
    return {"read_s": t1 - t0, "faces_s": t2 - t1, "model_s": t3 - t2, "total_s": t3 - t0,
            "frames_read": len(frames), "faces": len(crops), "score": float(probs.mean()) if len(probs) else np.nan}


speed_rows = []
for setup, device, gpu_detector in SETUPS:
    clf = classifier if str(classifier.device) == device else Classifier(APP_DIR / "models" / "deepfake_convnext_tiny.pt", device)
    det = Detector(DETECTOR_FILE, clf.min_face_score, on_gpu=gpu_detector)
    for name, path in SPEED_VIDEOS.items():
        props = video_props(path)
        for num_frames in FRAME_COUNTS:
            timed_run(path, det, clf, num_frames)  # warm-up
            runs = pd.DataFrame([timed_run(path, det, clf, num_frames) for _ in range(3)])
            speed_rows.append({"setup": setup, "model_device": device, "detector_device": det.device, "video": name,
                               **props, "frames_scored": num_frames, **runs.median().to_dict()})
            print(f"{setup} | {name} | {num_frames} frames | {speed_rows[-1]['total_s']:.2f} s")

speed = pd.DataFrame(speed_rows)
speed.to_csv(OUT / "speed.csv", index=False)
(OUT / "hardware.json").write_text(json.dumps(HARDWARE, indent=2))
print(speed.pivot_table(index=["setup", "video"], columns="frames_scored", values="total_s").round(2))

# %%
shutil.rmtree(SPEED_DIR)  # re-encoded test videos are not needed in the results
archive = shutil.make_archive(str(OUT), "zip", OUT)
print("Results:", archive)
