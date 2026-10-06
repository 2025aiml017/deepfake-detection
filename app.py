"""Streamlit demo: upload a video and get the capstone detector's REAL / FAKE decision.

    streamlit run app.py
"""
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from inference import MODEL_PATH, DeepfakeClassifier, FaceDetector, VideoResult, analyse_video, ensure_detector

VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]

st.set_page_config(page_title="Deepfake Video Detector", page_icon="🕵️")


@st.cache_resource(show_spinner="Loading the models. The first start also downloads the face detector (about 290 MB)…")
def load_models() -> tuple[FaceDetector, DeepfakeClassifier]:
    classifier = DeepfakeClassifier(MODEL_PATH)
    return FaceDetector(ensure_detector(), classifier.min_face_score), classifier


def run_analysis(uploaded, num_frames: int) -> VideoResult | None:
    bar = st.progress(0.0, text="Reading the video")
    try:
        # OpenCV reads from a path, so the upload goes to a temporary file first.
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp.flush()
            return analyse_video(Path(tmp.name), detector, classifier, num_frames,
                                 progress=lambda fraction, message: bar.progress(fraction, text=message))
    except ValueError as error:
        st.error(f"Could not read this video. {error} Try converting it to MP4 (H.264).")
        return None
    finally:
        bar.empty()


def score_chart(result: VideoResult) -> alt.LayerChart:
    fps = result.info.fps
    axis = "Time in video (s)" if fps > 0 else "Frame"
    faces = pd.DataFrame({"x": [i / fps if fps > 0 else i for i in result.frame_indices], "p_fake": result.probs})
    line = alt.Chart(faces).mark_line(point=True).encode(
        x=alt.X("x:Q", title=axis),
        y=alt.Y("p_fake:Q", title="P(fake) of the face", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("x:Q", title=axis, format=".1f"), alt.Tooltip("p_fake:Q", title="P(fake)", format=".3f")],
    )
    threshold = alt.Chart(pd.DataFrame({"y": [result.threshold]})).mark_rule(strokeDash=[6, 4], color="gray").encode(
        y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Threshold", format=".3f")])
    return line + threshold


detector, classifier = load_models()

with st.sidebar:
    st.header("Settings")
    num_frames = st.slider("Frames to analyse", min_value=8, max_value=64, value=32, step=8,
                           help="Evenly spaced frames, skipping the first and last 5% of the video. "
                                "More frames give a steadier score but take longer.")
    st.header("How it works")
    st.markdown(
        "1. **Faces:** RetinaFace finds the largest face in each sampled frame and aligns it on the eyes.\n"
        "2. **Per face:** a ConvNeXt-Tiny model fine-tuned on Celeb-DF v2 gives each face a probability "
        "of being fake.\n"
        f"3. **Decision:** the video's fake score is the average over its faces. The video is **FAKE** "
        f"if the score is at least **{classifier.threshold:.3f}**, a threshold chosen on validation videos.\n\n"
        "On 809 held-out videos of people the model never saw in training, it was right on 98.9%: "
        "all 105 real videos and 695 of 704 fakes."
    )
    st.caption(
        "Limitations: the model was trained only on Celeb-DF v2 face swaps of celebrity interviews. Other "
        "manipulations (lip-sync, fully generated video, filters), heavy compression or unusual footage can "
        "fool it. Treat the result as a research demo, not proof."
    )

st.title("Deepfake Video Detector")
st.write("Upload a short video of a person's face. The app checks sampled frames for signs of face swapping "
         "and gives one decision for the whole video.")

uploaded = st.file_uploader("Video", type=VIDEO_TYPES, max_upload_size=100,
                            help="MP4, MOV, AVI, MKV, WebM or M4V, up to 100 MB. Short clips (under a minute) work best. "
                                 "The preview plays only formats your browser supports (such as H.264 MP4); "
                                 "every format is still analysed.")
if uploaded is None:
    st.stop()

st.video(uploaded)

key = (uploaded.file_id, num_frames)
if st.session_state.get("result_key") != key:
    st.session_state.result = run_analysis(uploaded, num_frames)
    st.session_state.result_key = key
result: VideoResult | None = st.session_state.result
if result is None:
    st.stop()

if result.verdict == "NO FACE":
    st.warning(f"No face was found in the {result.frames_read} sampled frames, so there is nothing to judge. "
               "Try a video where one face is clearly visible.", title="No face found")
    st.stop()

n_faces = len(result.probs)
n_fake = sum(p >= result.threshold for p in result.probs)
if result.verdict == "FAKE":
    st.error(f"The faces in this video look manipulated. {n_fake} of {n_faces} faces scored above the threshold.",
             title="FAKE", icon=":material/gpp_bad:")
else:
    st.success(f"No sign of face swapping. {n_faces - n_fake} of {n_faces} faces scored below the threshold.",
               title="REAL", icon=":material/verified_user:")

fake_score, threshold_col, faces_col, time_col = st.columns(4)
fake_score.metric("Fake score", f"{result.score:.3f}", help="Average P(fake) over the detected faces. It is a "
                  "model score, not a calibrated probability.")
threshold_col.metric("Threshold", f"{result.threshold:.3f}", help="FAKE when the fake score is at or above this.")
faces_col.metric("Faces found", f"{n_faces} / {result.frames_read}", help="Frames where a face was detected.")
time_col.metric("Analysis time", f"{result.seconds:.1f} s")

st.subheader("Score of each face")
st.altair_chart(score_chart(result), width="stretch")
st.caption("Each point is one sampled frame. The dashed line is the decision threshold.")

with st.expander(f"The {n_faces} aligned faces the model scored", expanded=False):
    st.image(result.crops, width=96, caption=[f"P(fake) {p:.2f}" for p in result.probs])

info = result.info
details = [f"{info.width}×{info.height}", f"{info.fps:.0f} fps" if info.fps > 0 else None,
           f"{info.duration:.1f} s" if info.duration else None, f"model: {classifier.name}"]
st.caption(" · ".join(d for d in details if d))
