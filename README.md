# Deepfake Video Detection

Capstone project: classify a video as REAL or FAKE. RetinaFace finds and aligns the face in sampled frames, a
ConvNeXt-Tiny classifier fine-tuned on Celeb-DF v2 scores each face, and the video's fake score is the average.

| File | What it is |
|---|---|
| `docs/Deepfake_Detection_Capstone_Report.pdf` (and `.docx`) | Project report |
| `docs/Deepfake_Detection_Capstone_Presentation.pptx` | Presentation |
| `notebooks/training_and_evaluation.ipynb` | Final Kaggle notebook: data, training of seven models, evaluation, Run A and Run B |
| `notebooks/requirements.txt` | Packages used by the training notebook |
| `app.py`, `inference.py` | Streamlit demo app |
| `models/deepfake_convnext_tiny.pt` | Model used by the app |
| `scripts/make_app_bundle.py` | How the app's model file was built from the notebook's export |
| `experiments/lowres_speed.ipynb` | Low-resolution and speed tests (run on Kaggle) |

## Demo app

Upload a video and the app shows the decision, the fake score, a per-frame score chart and the aligned faces it
scored.

### Run locally

With Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first start downloads insightface's face-detector pack (about 290 MB) and keeps only `det_10g.onnx`
(17 MB) in `models/`.

### Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), choose **Create app**, select the repository, branch
   `main` and main file `app.py`.
3. Under **Advanced settings**, choose Python **3.12**.
4. Deploy. The first build takes a few minutes; `requirements.txt` installs the CPU-only PyTorch build.

Analysing 32 frames takes about 5 seconds on a laptop and longer on Community Cloud's shared CPUs.

## The model in the app

- **Checkpoint:** ConvNeXt-Tiny from the Google Colab run of the notebook (`runs/convnext_tiny_s42`), the
  checkpoint exported by section 10 as `deepfake_convnext_tiny_final.pt`. The Kaggle run described in the report
  retrained the same pipeline, so its numbers differ slightly from the ones below.
- **Threshold 0.495:** the notebook picks Run A's threshold with Youden's J on validation *frames* (0.0233 for
  this checkpoint) but applies it to video scores, which labels many real videos FAKE. The app uses Youden's J
  on validation *videos*, as the report's section 5.5 describes and Run B does.
- **Storage:** conv and linear weights are stored as float16 to keep the file under GitHub's 100 MB limit; the
  app runs in float32. On test faces this changed P(fake) by less than 0.0002.

Results of this checkpoint at each threshold:

| Threshold | Holdout accuracy (809 videos) | Real holdout videos called FAKE | Official test accuracy (518 videos) | Real test videos called FAKE |
|---|---|---|---|---|
| 0.0233 (validation frames) | 98.15% | 15 of 105 | 93.63% | 33 of 178 |
| **0.4952 (validation videos)** | **98.89%** | **0 of 105** | **99.23%** | **3 of 178** |

## Limitations

The model was trained only on Celeb-DF v2, which contains face swaps of celebrity interviews. It is not tested on
other manipulations (lip-sync, fully generated video, filters), on heavily compressed footage or on other
datasets, and it scores only the largest face in each frame. The insightface detector weights are licensed for
non-commercial research use.
