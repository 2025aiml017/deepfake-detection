"""Build the app's reference file (models/deepfake_convnext_tiny.json) from the training run's saved predictions.

    unzip -q runs-*.zip 'runs/*.csv' 'runs/*.json'
    python scripts/make_reference.py runs/

The file sits beside the model and holds what the app needs on top of the checkpoint:
- the Grad-CAM layer of the model;
- Platt scaling of the video score, fitted on validation videos with real and fake weighted equally, so the
  calibrated P(fake) assumes neither class is more common. It is kept only if it beats the raw score on holdout;
- the inconclusive range: the scores where both real and fake validation videos fell;
- evaluation data for the charts: holdout video scores, ROC curves, confusion counts at the app's threshold and the
  comparison of all trained models.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from make_app_bundle import OUTPUT as MODEL_PATH, video_youden_threshold

OUTPUT = MODEL_PATH.with_suffix(".json")
# Notebook section 1, ModelSpec.cam_layer: the last feature block of each model.
CAM_LAYERS = {
    "efficientnet_b0": "bn2", "resnet50": "layer4", "xception": "act4", "convnext_tiny": "stages.3",
    "swin_tiny": "layers.3", "efficientnet_b4": "bn2", "clip_vit_b16": "blocks.11.norm1",
}
LOGIT_CLIP = 1e-4  # video scores of exactly 0 or 1 are common


def video_scores(preds_csv: Path) -> pd.DataFrame:
    """Mean frame P(fake) per video, as the notebook and the app score a video."""
    frames = pd.read_csv(preds_csv)
    return frames.groupby("stem").agg(label=("label", "first"), score=("prob", "mean"))


def logit(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(scores / (1 - scores))


def class_weights(labels: np.ndarray) -> np.ndarray:
    """Weights that give real and fake videos equal total weight."""
    return np.where(labels == 1, 0.5 / np.mean(labels == 1), 0.5 / np.mean(labels == 0))


def fit_platt(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of P(fake) = sigmoid(slope * logit(score) + intercept), by Newton's method."""
    x, weights, params = logit(scores), class_weights(labels), np.array([1.0, 0.0])
    design = np.stack([x, np.ones_like(x)], axis=1)
    for _ in range(100):
        probs = 1 / (1 + np.exp(-design @ params))
        gradient = design.T @ (weights * (probs - labels))
        hessian = design.T @ (design * (weights * probs * (1 - probs))[:, None])
        step = np.linalg.solve(hessian, gradient)
        params -= step
        if np.abs(step).max() < 1e-10:
            break
    return float(params[0]), float(params[1])


def calibrate(scores: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(slope * logit(scores) + intercept)))


def balanced_log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    probs, weights = np.clip(probs, 1e-6, 1 - 1e-6), class_weights(labels)
    losses = -(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return float(np.sum(weights * losses) / np.sum(weights))


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """False and true positive rates at every distinct score, keeping only the corners of the curve."""
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    last_of_each = np.r_[np.where(np.diff(scores))[0], len(scores) - 1]
    tps, fps = np.cumsum(labels)[last_of_each], np.cumsum(1 - labels)[last_of_each]
    corners = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
    return np.r_[0, fps[corners] / fps[-1]], np.r_[0, tps[corners] / tps[-1]]


def evaluate(videos: pd.DataFrame, threshold: float, inconclusive: tuple[float, float] | None,
             calibration: tuple[float, float]) -> dict:
    scores, labels = videos.score.to_numpy(), videos.label.to_numpy()
    called_fake, wrong = scores >= threshold, (scores >= threshold) != (labels == 1)
    flagged = (scores >= inconclusive[0]) & (scores <= inconclusive[1]) if inconclusive else np.zeros_like(wrong)
    fpr, tpr = roc_curve(scores, labels)
    return {
        "videos": len(videos),
        "tn": int(np.sum(~called_fake & (labels == 0))), "fp": int(np.sum(called_fake & (labels == 0))),
        "fn": int(np.sum(~called_fake & (labels == 1))), "tp": int(np.sum(called_fake & (labels == 1))),
        "inconclusive_videos": int(flagged.sum()), "inconclusive_errors": int((flagged & wrong).sum()),
        "auc": float(np.trapezoid(tpr, fpr)),
        "roc": {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()},
        "log_loss_raw": balanced_log_loss(labels, scores),
        "log_loss_calibrated": balanced_log_loss(labels, calibrate(scores, *calibration)),
    }


def model_comparison(runs: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(runs.glob("*/metrics.json")):
        metrics = json.loads(metrics_path.read_text())
        robustness = pd.read_csv(metrics_path.parent / "robustness.csv").set_index("condition").video_auc
        degraded = robustness.drop("clean")
        rows.append({
            "model": metrics["model"], "params_m": metrics["params_m"], "gpu_img_per_s": metrics["gpu_img_per_s"],
            "holdout_video_auc": metrics["holdout_video_auc"], "holdout_video_eer": metrics["holdout_video_eer"],
            "robustness_clean_auc": float(robustness["clean"]),
            "robustness_worst_auc": float(degraded.min()), "robustness_worst_condition": degraded.idxmin(),
        })
    return sorted(rows, key=lambda row: -row["holdout_video_auc"])


def rounded(value, digits: int = 6):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, help="the notebook's runs/ folder, with each model's CSV and JSON files")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="the app's model file")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    bundle = torch.load(args.model, map_location="cpu", weights_only=True)
    run = args.runs / f"{bundle['model_key']}_s42"
    threshold = float(bundle["threshold"])
    if not math.isclose(video_youden_threshold(run / "preds_val.csv"), threshold, abs_tol=1e-6):
        raise SystemExit(f"{run} is not the run that produced {args.model}: its validation threshold differs")

    val, holdout, test = (video_scores(run / f"preds_{split}.csv") for split in ("val", "holdout", "test"))
    real, fake = val.score[val.label == 0], val.score[val.label == 1]
    inconclusive = (float(fake.min()), float(real.max())) if fake.min() <= real.max() else None
    calibration = fit_platt(val.score.to_numpy(), val.label.to_numpy())
    evaluation = {name: evaluate(videos, threshold, inconclusive, calibration)
                  for name, videos in (("holdout", holdout), ("test", test))}
    keep_calibration = evaluation["holdout"]["log_loss_calibrated"] < evaluation["holdout"]["log_loss_raw"]
    evaluation["holdout"]["real_scores"] = holdout.score[holdout.label == 0].round(4).tolist()
    evaluation["holdout"]["fake_scores"] = holdout.score[holdout.label == 1].round(4).tolist()

    reference = {
        "model": bundle["model_key"],
        "threshold": threshold,
        "cam_layer": CAM_LAYERS[bundle["model_key"]],
        "calibration": {"slope": calibration[0], "intercept": calibration[1], "logit_clip": LOGIT_CLIP}
        if keep_calibration else None,
        "inconclusive": list(inconclusive) if inconclusive else None,
        "evaluation": evaluation,
        "models": model_comparison(args.runs),
    }
    args.output.write_text(json.dumps(rounded(reference), indent=1) + "\n")

    print(f"inconclusive range {inconclusive} | calibration slope {calibration[0]:.4f} intercept {calibration[1]:.4f}")
    for name, result in evaluation.items():
        print(f"{name}: inconclusive {result['inconclusive_videos']} videos, {result['inconclusive_errors']} of "
              f"{result['fp'] + result['fn']} errors | balanced log loss raw {result['log_loss_raw']:.4f} "
              f"calibrated {result['log_loss_calibrated']:.4f} | AUC {result['auc']:.5f}")
    print(f"calibration {'kept' if keep_calibration else 'dropped: it does not beat the raw score on holdout'}")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
