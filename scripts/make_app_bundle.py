"""Build the app's model file (models/deepfake_convnext_tiny.pt) from the notebook's exported bundle.

    python scripts/make_app_bundle.py deepfake_convnext_tiny_final.pt [--preds-val runs/convnext_tiny_s42/preds_val.csv]

Two changes, everything else is copied unchanged:
- Conv/linear weights are stored as float16 (the app casts them back to float32), which halves the file to
  about 56 MB so it fits in plain git. Biases, norms and layer scales stay float32.
- The threshold is replaced by the video-level one. The notebook chose its threshold with Youden's J on
  validation *frames* (0.0233 for this checkpoint) but applies it to video scores; the app uses Youden's J on
  validation *videos* (mean frame P(fake)), as report section 5.5 describes. With --preds-val it is recomputed
  from that run's saved validation predictions, otherwise the value computed from them is used.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Video-level Youden threshold from runs/convnext_tiny_s42/preds_val.csv of the run that produced this checkpoint.
VIDEO_THRESHOLD = 0.4951908
OUTPUT = Path(__file__).resolve().parent.parent / "models" / "deepfake_convnext_tiny.pt"


def video_youden_threshold(preds_csv: Path) -> float:
    """Youden's J over validation video scores; same choice as sklearn's roc_curve + argmax in the notebook."""
    probs, labels = defaultdict(list), {}
    with open(preds_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            probs[row["stem"]].append(float(row["prob"]))
            labels[row["stem"]] = int(row["label"])
    scores = np.array([np.mean(p) for p in probs.values()])
    fake = np.array([labels[stem] for stem in probs]) == 1
    candidates = np.unique(scores)[::-1]  # highest first, so ties keep the highest threshold like np.argmax
    j = [(scores[fake] >= t).mean() - (scores[~fake] >= t).mean() for t in candidates]
    return float(candidates[int(np.argmax(j))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bundle", type=Path, help="bundle exported by the notebook (section 10)")
    parser.add_argument("--preds-val", type=Path, help="preds_val.csv from the same run, to recompute the threshold")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    threshold = video_youden_threshold(args.preds_val) if args.preds_val else VIDEO_THRESHOLD
    compact = {
        **bundle,
        "state_dict": {name: tensor.half() if tensor.ndim >= 2 else tensor
                       for name, tensor in bundle["state_dict"].items()},
        "threshold": threshold,
        "frame_threshold": bundle["threshold"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(f"threshold {bundle['threshold']:.4f} (frames) -> {threshold:.4f} (videos) | wrote {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
