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
| `models/deepfake_convnext_tiny.json` | The model's reference file: Grad-CAM layer, calibration, inconclusive range and the evaluation data the app's charts show |
| `scripts/make_app_bundle.py` | How the app's model file was built from the notebook's export |
| `scripts/make_reference.py` | How the reference file was built from the training run's predictions |
| `experiments/lowres_speed.ipynb` | Low-resolution and speed tests (run on Kaggle) |

## Demo app

The **Analyse a video** tab takes an upload and shows:

- the decision (REAL, FAKE, or INCONCLUSIVE when the score falls where real and fake validation videos overlap),
  the fake score and the calibrated P(fake);
- where the score falls among the 809 holdout videos;
- Grad-CAM heatmaps of the four highest-scoring faces, showing which parts of each face pushed its score towards
  FAKE;
- the score of each face over the video, with a rolling average and the frames where no face was found;
- every aligned face it scored, with heatmaps on request.

The **About the model** tab shows its accuracy and ROC curves on the holdout and official test videos, how the
inconclusive range and calibration were chosen, and why ConvNeXt-Tiny was picked over the other six models.

Browsers play only some video formats: MP4, MOV and M4V with H.264, and WebM with VP8 or VP9. For anything else,
including the MPEG-4 Part 2 of the Celeb-DF clips, the app plays a 480p H.264 copy made with the FFmpeg build in
`imageio-ffmpeg`. The copy is only for the player; the analysis always reads the original file, because
re-encoding changes the fake score (see Limitations).

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

Analysing 32 frames takes about 5 seconds on a laptop and longer on Community Cloud's shared CPUs. The four
heatmaps add about a second, and heatmaps for all 32 faces about 7 more.

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

The reference file adds three things on top of the threshold, all chosen on the same run's validation videos:

- **Inconclusive range 0.240 to 0.426:** fake validation videos scored as low as 0.240 and real ones as high as
  0.426, so a score in between is labelled inconclusive. The REAL / FAKE decision is unchanged, so the table above
  still holds. On holdout the range flags 6 videos, including 5 of the 9 wrong calls; on the official test split
  it flags 3 videos and none of its 4 wrong calls.
- **Calibrated P(fake):** Platt scaling of the video score, with real and fake videos weighted equally so it does
  not assume most uploads are fake. It lowers the log loss on holdout (0.042 to 0.018) but raises it on the
  official test split (0.031 to 0.053), where some real videos score higher than any real validation video.
- **Grad-CAM layer:** `stages.3`, as in the notebook's section 9. The app's heatmaps match the notebook's to
  within 1e-6.

To rebuild it, unzip the run's CSV and JSON files (the checkpoints are not needed) and run the script:

```bash
unzip -q runs-*.zip 'runs/*.csv' 'runs/*.json'
python scripts/make_reference.py runs/
```

## Limitations

The model was trained only on Celeb-DF v2, which contains face swaps of celebrity interviews. It is not tested on
other manipulations (lip-sync, fully generated video, filters), on heavily compressed footage or on other
datasets, and it scores only the largest face in each frame. Ordinary video re-encoding can hide a fake: in a
check on one fake Celeb-DF clip (id49_id50_0000), re-encoding it as H.264 dropped its fake score from 0.853 to
0.787 at CRF 18, 0.390 (inconclusive) at FFmpeg's default CRF 23 and 0.001 (REAL) at CRF 28. The notebook's
robustness test only re-compressed single frames as JPEG. The inconclusive range and calibration come from
112 real validation videos scored on about 8 frames each, so they are rough. Grad-CAM heatmaps show where the
model looked, not proof of manipulation. The insightface detector weights are licensed for non-commercial
research use.
