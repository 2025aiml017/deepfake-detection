"""Streamlit demo: upload a video and get the capstone detector's REAL / FAKE decision.

    streamlit run app.py
"""
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path

import altair as alt
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st

from inference import (MODEL_PATH, DeepfakeClassifier, FaceDetector, VideoResult, analyse_video, ensure_detector,
                       heatmap_overlay)

VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v"]
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# Formats every current browser plays as they are: extension -> (video codecs, audio codecs), with 8-bit 4:2:0
# video. Anything else (Celeb-DF's MPEG-4 Part 2, AVI, MKV, HEVC, 10-bit, PCM audio, ...) gets an H.264 preview.
BROWSER_FORMATS = {
    ".mp4": ({"h264"}, {"aac", "mp3"}), ".m4v": ({"h264"}, {"aac", "mp3"}), ".mov": ({"h264"}, {"aac", "mp3"}),
    ".webm": ({"vp8", "vp9"}, {"opus", "vorbis"}),
}
PREVIEW_HEIGHT = 480
RESULTS_KEPT = 4  # analysed videos kept per session, so re-uploads and earlier slider values are instant
TOP_FACES = 4  # highest-scoring faces shown with a heatmap straight away
ROLLING_FACES = 5
# Blue for the real side and orange for the fake side (slots 1 and 2 of the dataviz reference palette, stepped for
# each theme), grey for reference marks, and the theme's surface for the ring around dots.
COLORS = {
    "light": {"real": "#2a78d6", "fake": "#eb6834", "muted": "#898781", "ink": "#52514e", "surface": "#ffffff"},
    "dark": {"real": "#3987e5", "fake": "#d95926", "muted": "#898781", "ink": "#c3c2b7", "surface": "#0e1117"},
}
DEGRADATIONS = {"jpeg_q50": "JPEG quality 50", "jpeg_q30": "JPEG quality 30", "downscale_x0.5": "half resolution",
                "blur_r1.5": "blur"}

st.set_page_config(page_title="Deepfake Video Detector", page_icon="🕵️")


@st.cache_resource(show_spinner="Loading the models. The first start also downloads the face detector (about 290 MB)…")
def load_models() -> tuple[FaceDetector, DeepfakeClassifier]:
    classifier = DeepfakeClassifier(MODEL_PATH)
    return FaceDetector(ensure_detector(), classifier.min_face_score), classifier


def colors() -> dict[str, str]:
    return COLORS["dark" if st.context.theme.type == "dark" else "light"]


def percent(p: float) -> str:
    return "<1%" if p < 0.01 else ">99%" if p > 0.99 else f"{p:.0%}"


def file_digest(uploaded) -> str:
    """SHA-256 of an upload, computed once per upload."""
    digests = st.session_state.setdefault("digests", {})
    if uploaded.file_id not in digests:
        digests[uploaded.file_id] = hashlib.sha256(uploaded.getbuffer()).hexdigest()
    return digests[uploaded.file_id]


def browser_plays(path: Path) -> bool:
    """Whether every browser plays this file as it is. `ffmpeg -i` with no output file lists the streams (and exits
    with an error, as expected)."""
    if path.suffix.lower() not in BROWSER_FORMATS:
        return False
    video_codecs, audio_codecs = BROWSER_FORMATS[path.suffix.lower()]
    streams = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                             errors="replace").stderr
    videos = re.findall(r"Stream #.*?: Video: (\w+)(.*)", streams)
    audios = re.findall(r"Stream #.*?: Audio: (\w+)", streams)
    return (bool(videos) and all(codec in video_codecs and re.search(r"\byuvj?420p\b", rest) for codec, rest in videos)
            and all(codec in audio_codecs for codec in audios))


def make_preview(uploaded) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        source, target = Path(folder) / f"upload{Path(uploaded.name).suffix}", Path(folder) / "preview.mp4"
        source.write_bytes(uploaded.getbuffer())
        if browser_plays(source):
            return {"video": None, "note": None}
        try:
            with st.spinner("Converting the video so your browser can play it…"):
                subprocess.run([FFMPEG, "-v", "error", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                                "-vf", f"scale=-2:'2*trunc(min({PREVIEW_HEIGHT},ih)/2)'", "-c:a", "aac",
                                "-movflags", "+faststart", str(target)], check=True, capture_output=True, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return {"video": None, "note": "Browsers may not play this video's format, and converting it for the "
                                           "preview failed. The analysis still uses the original file."}
        return {"video": target.read_bytes(),
                "note": "Browsers can't reliably play this video's format, so this preview is a converted copy. The "
                        "analysis uses the original file, as re-encoding changes the fake score."}


def browser_preview(uploaded) -> dict:
    """What the video player shows: the upload itself when every browser plays it, otherwise an H.264 copy made only
    for the player. Kept in the session by file content, like the results."""
    previews = st.session_state.setdefault("previews", {})
    digest = file_digest(uploaded)
    if digest not in previews:
        previews[digest] = make_preview(uploaded)
        while len(previews) > RESULTS_KEPT:
            previews.pop(next(iter(previews)))
    return previews[digest]


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


def analysis(uploaded, num_frames: int) -> dict:
    """The result for this video and frame count, and the heatmaps computed for it so far. Kept in the session (not
    a cache shared between users), keyed by the file's content, so uploading the same video again reuses it."""
    results = st.session_state.setdefault("results", {})
    key = (file_digest(uploaded), num_frames)
    if key not in results:
        results[key] = {"result": run_analysis(uploaded, num_frames), "heatmaps": {}}
        while len(results) > RESULTS_KEPT:
            results.pop(next(iter(results)))
    return results[key]


def heatmaps(entry: dict, faces: list[int]) -> dict[int, np.ndarray]:
    """Grad-CAM overlays for these face indices, computing only the ones not already in the session."""
    cache, crops = entry["heatmaps"], entry["result"].crops
    missing = [i for i in faces if i not in cache]
    if missing:
        with st.spinner(f"Computing {len(missing)} heatmaps…"):
            for i, heatmap in zip(missing, classifier.grad_cam([crops[i] for i in missing])):
                cache[i] = heatmap_overlay(crops[i], heatmap)
    return {i: cache[i] for i in faces}


def face_caption(result: VideoResult, i: int) -> str:
    fps, frame = result.info.fps, result.frame_indices[i]
    return f"P(fake) {result.probs[i]:.2f} · " + (f"{frame / fps:.1f} s" if fps > 0 else f"frame {frame}")


def score_context_chart(result: VideoResult) -> alt.LayerChart:
    """This video's score on one axis with every holdout video's score, the threshold and the inconclusive range."""
    palette, holdout = colors(), classifier.reference["evaluation"]["holdout"]
    rows = ["This video", "Real holdout videos", "Fake holdout videos"]
    videos = pd.DataFrame([(group, score) for group, scores in zip(rows[1:], (holdout["real_scores"],
                                                                              holdout["fake_scores"]))
                           for score in scores], columns=["group", "score"])
    videos["jitter"] = np.random.default_rng(0).uniform(0.15, 0.85, len(videos))
    x = alt.X("score:Q", title="Video fake score", scale=alt.Scale(domain=[0, 1], padding=8, nice=False))
    y = alt.Y("group:N", title=None, sort=rows, axis=alt.Axis(labelLimit=200))

    layers = []
    if classifier.inconclusive_range:
        low, high = classifier.inconclusive_range
        band = pd.DataFrame({"low": [low], "high": [high], "label": [f"Inconclusive: {low:.2f} to {high:.2f}"]})
        layers.append(alt.Chart(band).mark_rect(color=palette["muted"], opacity=0.18).encode(
            x="low:Q", x2="high:Q", tooltip=[alt.Tooltip("label:N", title="Range")]))
    layers.append(alt.Chart(pd.DataFrame({"t": [result.threshold]})).mark_rule(
        strokeDash=[6, 4], color=palette["muted"]).encode(x="t:Q", tooltip=[alt.Tooltip("t:Q", title="Threshold",
                                                                                         format=".3f")]))
    layers.append(alt.Chart(videos).mark_circle(size=28, opacity=0.4).encode(
        x=x, y=y, yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("group:N", scale=alt.Scale(domain=rows[1:], range=[palette["real"], palette["fake"]]),
                        legend=None),
        tooltip=[alt.Tooltip("group:N", title="Video"), alt.Tooltip("score:Q", title="Fake score", format=".3f")]))
    this = pd.DataFrame({"group": [rows[0]], "score": [result.score]})
    layers.append(alt.Chart(this).mark_rule(color=palette["ink"], strokeWidth=2).encode(x="score:Q"))
    layers.append(alt.Chart(this).mark_point(shape="diamond", size=220, filled=True, color=palette["ink"],
                                             stroke=palette["surface"], strokeWidth=2).encode(
        x=x, y=y, tooltip=[alt.Tooltip("score:Q", title="This video's fake score", format=".3f")]))
    return alt.layer(*layers).properties(height=alt.Step(44))


def score_chart(result: VideoResult) -> alt.LayerChart:
    """P(fake) of each face over the video, with a rolling average and the frames where no face was found."""
    palette, fps = colors(), result.info.fps
    axis = "Time in video (s)" if fps > 0 else "Frame"
    position = (lambda frame: frame / fps) if fps > 0 else (lambda frame: frame)
    faces = pd.DataFrame({"x": [position(i) for i in result.frame_indices], "p_fake": result.probs}).sort_values("x")
    faces["side"] = np.where(faces.p_fake >= result.threshold, "At or above threshold", "Below threshold")
    faces["rolling"] = faces.p_fake.rolling(ROLLING_FACES, center=True, min_periods=1).mean()
    x = alt.X("x:Q", title=axis)
    y = alt.Y("p_fake:Q", title="P(fake) of the face", scale=alt.Scale(domain=[0, 1]))

    layers = []
    if result.no_face_frames:
        missing = pd.DataFrame({"x": [position(i) for i in result.no_face_frames], "label": "No face found"})
        layers.append(alt.Chart(missing).mark_rule(color=palette["muted"], opacity=0.25, strokeWidth=8).encode(
            x="x:Q", tooltip=[alt.Tooltip("label:N", title="Frame"), alt.Tooltip("x:Q", title=axis, format=".1f")]))
    layers.append(alt.Chart(pd.DataFrame({"y": [result.threshold]})).mark_rule(
        strokeDash=[6, 4], color=palette["muted"]).encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Threshold",
                                                                                         format=".3f")]))
    layers.append(alt.Chart(faces).mark_line(color=palette["muted"], strokeWidth=1).encode(x=x, y=y))
    layers.append(alt.Chart(faces).mark_line(color=palette["ink"], strokeWidth=2).encode(
        x=x, y="rolling:Q", tooltip=[alt.Tooltip("x:Q", title=axis, format=".1f"),
                                     alt.Tooltip("rolling:Q", title=f"Average of {ROLLING_FACES} faces",
                                                 format=".3f")]))
    layers.append(alt.Chart(faces).mark_circle(size=70, opacity=1, stroke=palette["surface"], strokeWidth=1.5).encode(
        x=x, y=y, color=alt.Color("side:N", title=None, legend=alt.Legend(orient="top"),
                                  scale=alt.Scale(domain=["Below threshold", "At or above threshold"],
                                                  range=[palette["real"], palette["fake"]])),
        tooltip=[alt.Tooltip("x:Q", title=axis, format=".1f"), alt.Tooltip("p_fake:Q", title="P(fake)", format=".3f")]))
    return alt.layer(*layers)


def roc_chart(split: dict) -> alt.LayerChart:
    """Video-level ROC curve, zoomed to the top-left corner, with the app's threshold marked."""
    palette = colors()
    x = alt.X("fpr:Q", title="Share of real videos called FAKE",
              scale=alt.Scale(domain=[0, 0.1], padding=6, nice=False))
    y = alt.Y("tpr:Q", title="Share of fake videos caught",
              scale=alt.Scale(domain=[0.9, 1], padding=6, nice=False))
    point = pd.DataFrame({"fpr": [split["fp"] / (split["fp"] + split["tn"])],
                          "tpr": [split["tp"] / (split["tp"] + split["fn"])]})
    curve = alt.Chart(pd.DataFrame(split["roc"])).mark_line(color=palette["ink"], strokeWidth=2, clip=True).encode(
        x=x, y=y)
    threshold = alt.Chart(point).mark_circle(size=90, opacity=1, color=palette["fake"], stroke=palette["surface"],
                                             strokeWidth=2).encode(
        x=x, y=y, tooltip=[alt.Tooltip("fpr:Q", title="False positive rate", format=".3f"),
                           alt.Tooltip("tpr:Q", title="True positive rate", format=".3f")])
    return (curve + threshold).properties(height=260)


def model_chart(models: pd.DataFrame) -> alt.LayerChart:
    """Each trained model's holdout AUC on clean faces and under its worst degradation."""
    palette = colors()
    conditions = {"robustness_clean_auc": "Clean faces", "robustness_worst_auc": "Worst degradation"}
    points = models.melt(id_vars=["name", "worst"], value_vars=list(conditions), var_name="condition",
                         value_name="auc")
    points["condition"] = points.condition.map(conditions)
    x = alt.X("auc:Q", title="Holdout video AUC", scale=alt.Scale(domain=[0.97, 1]))
    y = alt.Y("name:N", title=None, sort=list(models.name), axis=alt.Axis(labelLimit=220))
    span = alt.Chart(models).mark_rule(color=palette["muted"], strokeWidth=2).encode(
        x=alt.X("robustness_worst_auc:Q", scale=alt.Scale(domain=[0.97, 1])), x2="robustness_clean_auc:Q", y=y)
    dots = alt.Chart(points).mark_point(shape="circle", size=90, strokeWidth=2, opacity=1,
                                        color=palette["ink"]).encode(
        x=x, y=y,
        fill=alt.Fill("condition:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=list(conditions.values()), range=[palette["ink"], palette["surface"]])),
        tooltip=[alt.Tooltip("name:N", title="Model"), alt.Tooltip("condition:N", title="Faces"),
                 alt.Tooltip("worst:N", title="Worst degradation"), alt.Tooltip("auc:Q", title="AUC", format=".4f")])
    return (span + dots).properties(height=alt.Step(34))


def show_verdict(result: VideoResult) -> None:
    n_faces = len(result.probs)
    n_fake = sum(p >= result.threshold for p in result.probs)
    if result.inconclusive:
        low, high = classifier.inconclusive_range
        st.warning(f"The model leans {result.verdict} (fake score {result.score:.3f}, threshold "
                   f"{result.threshold:.3f}), but both real and fake validation videos scored between {low:.2f} and "
                   f"{high:.2f}. Treat this result as uncertain.", title="INCONCLUSIVE", icon=":material/help:")
    elif result.verdict == "FAKE":
        st.error(f"The faces in this video look manipulated. {n_fake} of {n_faces} faces scored above the "
                 "threshold.", title="FAKE", icon=":material/gpp_bad:")
    else:
        st.success(f"No sign of face swapping. {n_faces - n_fake} of {n_faces} faces scored below the threshold.",
                   title="REAL", icon=":material/verified_user:")

    fake_score, calibrated, faces_col, time_col = st.columns(4)
    fake_score.metric("Fake score", f"{result.score:.3f}", help="Average P(fake) over the detected faces. The video "
                      f"is FAKE when this is at least {result.threshold:.3f}.")
    if result.calibrated is not None:
        calibrated.metric("Calibrated P(fake)", percent(result.calibrated),
                          help="The fake score as a probability: Platt scaling fitted on validation videos, with "
                               "real and fake videos taken as equally likely. Better calibrated than the raw score on "
                               "holdout videos, but over-confident on a few real test videos (see About the model).")
    else:
        calibrated.metric("Threshold", f"{result.threshold:.3f}", help="FAKE when the fake score is at or above this.")
    faces_col.metric("Faces found", f"{n_faces} / {result.frames_read}", help="Frames where a face was detected.")
    time_col.metric("Analysis time", f"{result.seconds:.1f} s")


def show_faces(entry: dict) -> None:
    result = entry["result"]
    n_faces = len(result.probs)
    if classifier.reference.get("cam_layer"):
        st.subheader("Where the model looked")
        top = sorted(range(n_faces), key=lambda i: result.probs[i], reverse=True)[:TOP_FACES]
        overlays = heatmaps(entry, top)
        for column, i in zip(st.columns(len(top)), top):
            column.image(result.crops[i], width="stretch", alt=f"Face, {face_caption(result, i)}")
            column.image(overlays[i], width="stretch", caption=face_caption(result, i),
                         alt=f"Heatmap, {face_caption(result, i)}")
        st.caption(f"The {len(top)} highest-scoring faces, each above its Grad-CAM heatmap. Lit, orange areas pushed "
                   "the score towards FAKE. Each heatmap is scaled to its own face, so a lit area on a low-scoring "
                   "face is not evidence of manipulation.")

    st.subheader("Score of each face")
    st.altair_chart(score_chart(result), width="stretch")
    st.caption(f"Each dot is one sampled frame; the thick line is the average of {ROLLING_FACES} neighbouring faces "
               "and the dashed line is the decision threshold."
               + (" Grey vertical lines mark sampled frames where no face was found." if result.no_face_frames
                  else ""))

    # Tracked state (key + rerun) keeps the expander open when the heatmap toggle inside it reruns the app.
    with st.expander(f"All {n_faces} aligned faces the model scored", key="all_faces", on_change="rerun"):
        show_heatmaps = classifier.reference.get("cam_layer") and st.toggle(
            "Show heatmaps", help="Computes a Grad-CAM heatmap for every face, which takes a few seconds.")
        images = list(heatmaps(entry, list(range(n_faces))).values()) if show_heatmaps else result.crops
        st.image(images, width=96, caption=[f"P(fake) {p:.2f}" for p in result.probs])


def show_analysis() -> None:
    uploaded = st.file_uploader("Video", type=VIDEO_TYPES, max_upload_size=100,
                                help="MP4, MOV, AVI, MKV, WebM or M4V, up to 100 MB. Short clips (under a minute) "
                                     "work best. Formats browsers can't play are converted for the preview only; "
                                     "the analysis always uses the original file.")
    if uploaded is None:
        return
    preview = browser_preview(uploaded)
    st.video(uploaded if preview["video"] is None else preview["video"])
    if preview["note"]:
        st.caption(preview["note"])

    entry = analysis(uploaded, num_frames)
    result: VideoResult | None = entry["result"]
    if result is None:
        return
    if result.verdict == "NO FACE":
        st.warning(f"No face was found in the {result.frames_read} sampled frames, so there is nothing to judge. "
                   "Try a video where one face is clearly visible.", title="No face found")
        return

    show_verdict(result)
    if classifier.reference:
        st.subheader("Where this score falls")
        st.altair_chart(score_context_chart(result), width="stretch")
        st.caption("Each dot is one holdout video, of people the model never saw in training. The dashed line is "
                   "the threshold and the shaded range is inconclusive.")
    show_faces(entry)

    info = result.info
    details = [f"{info.width}×{info.height}", f"{info.fps:.0f} fps" if info.fps > 0 else None,
               f"{info.duration:.1f} s" if info.duration else None, f"model: {classifier.name}"]
    st.caption(" · ".join(d for d in details if d))


def show_about() -> None:
    reference = classifier.reference
    if not reference:
        st.info("The model's reference file is missing; run scripts/make_reference.py to build it.")
        return
    holdout, test = reference["evaluation"]["holdout"], reference["evaluation"]["test"]

    st.subheader("Accuracy at the app's threshold")
    st.write(f"A video is FAKE when its fake score is at least {reference['threshold']:.3f}, the threshold that best "
             "separated real from fake validation videos. Holdout videos show people the model never saw in "
             "training; the official Celeb-DF test split shares people with the training set.")
    for column, (title, split) in zip(st.columns(2), (("Holdout", holdout), ("Official test", test))):
        accuracy = (split["tn"] + split["tp"]) / split["videos"]
        column.markdown(f"**{title}:** {split['videos']} videos, {accuracy:.2%} correct")
        column.table(pd.DataFrame({"Called REAL": [split["tn"], split["fn"]], "Called FAKE": [split["fp"], split["tp"]]},
                                  index=["Real videos", "Fake videos"]))
        column.altair_chart(roc_chart(split), width="stretch")
    st.caption(f"ROC curves, zoomed to the top-left corner: the curve shows every threshold, the orange dot the app's. "
               f"AUC {holdout['auc']:.4f} on holdout and {test['auc']:.4f} on the official test split.")

    st.subheader("Inconclusive results and calibrated P(fake)")
    lines = []
    if reference["inconclusive"]:
        low, high = reference["inconclusive"]
        lines.append(
            f"**Inconclusive range.** Real validation videos scored up to {high:.3f} and fake ones as low as "
            f"{low:.3f}, so a score in between is labelled inconclusive. On holdout it flagged "
            f"{holdout['inconclusive_videos']} of {holdout['videos']} videos, including {holdout['inconclusive_errors']} "
            f"of the {holdout['fp'] + holdout['fn']} wrong calls; on the official test split it flagged "
            f"{test['inconclusive_videos']} videos and {test['inconclusive_errors']} of its "
            f"{test['fp'] + test['fn']} wrong calls.")
    if reference["calibration"]:
        lines.append(
            f"**Calibrated P(fake).** Platt scaling fitted on validation videos, with real and fake videos weighted "
            f"equally so it does not assume most uploads are fake. On holdout it cut the log loss from "
            f"{holdout['log_loss_raw']:.3f} to {holdout['log_loss_calibrated']:.3f}. On the official test split it "
            f"rose from {test['log_loss_raw']:.3f} to {test['log_loss_calibrated']:.3f}: some real test videos "
            "scored higher than any real validation video, and calibration turns those into confident FAKE calls.")
    lines.append("These figures come from about 8 faces per video; the app averages 8 to 64, which makes a video's "
                 "score steadier.")
    st.markdown("\n\n".join(lines))

    st.subheader("Why ConvNeXt-Tiny")
    models = pd.DataFrame(reference["models"])
    models["name"] = [f"{m} (this app)" if m == reference["model"] else m for m in models.model]
    models["worst"] = models.robustness_worst_condition.map(DEGRADATIONS)
    st.write("Seven models were trained on the same faces. ConvNeXt-Tiny separated real from fake holdout videos best "
             "and lost the least when the faces were compressed, downscaled or blurred.")
    st.altair_chart(model_chart(models), width="stretch")
    st.caption("Holdout video AUC from up to 8 faces per video: filled dots on clean faces, hollow dots under the "
               "degradation that hurt each model most (JPEG quality 50 or 30, half resolution or blur).")
    st.dataframe(models[["name", "params_m", "gpu_img_per_s", "holdout_video_auc", "holdout_video_eer"]],
                 hide_index=True, column_config={
                     "name": "Model",
                     "params_m": st.column_config.NumberColumn("Parameters (M)", format="%.1f"),
                     "gpu_img_per_s": st.column_config.NumberColumn("Speed (faces/s, T4 GPU)", format="%.0f"),
                     "holdout_video_auc": st.column_config.NumberColumn("Holdout AUC", format="%.4f"),
                     "holdout_video_eer": st.column_config.NumberColumn("Holdout EER", format="%.4f"),
                 })


detector, classifier = load_models()

with st.sidebar:
    st.header("Settings")
    num_frames = st.slider("Frames to analyse", min_value=8, max_value=64, value=32, step=8,
                           help="Evenly spaced frames, skipping the first and last 5% of the video. "
                                "More frames give a steadier score but take longer.")
    st.header("How it works")
    inconclusive = ""
    if classifier.inconclusive_range:
        low, high = classifier.inconclusive_range
        inconclusive = (f" Scores from **{low:.2f}** to **{high:.2f}** are marked **inconclusive**: real and fake "
                        "validation videos both scored there.")
    st.markdown(
        "1. **Faces:** RetinaFace finds the largest face in each sampled frame and aligns it on the eyes.\n"
        "2. **Per face:** a ConvNeXt-Tiny model fine-tuned on Celeb-DF v2 gives each face a probability "
        "of being fake.\n"
        f"3. **Decision:** the video's fake score is the average over its faces. The video is **FAKE** "
        f"if the score is at least **{classifier.threshold:.3f}**, a threshold chosen on validation videos."
        f"{inconclusive}\n"
        "4. **Heatmaps:** Grad-CAM shows which parts of a face pushed its score towards FAKE.\n\n"
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

analyse_tab, about_tab = st.tabs(["Analyse a video", "About the model"])
with analyse_tab:
    show_analysis()
with about_tab:
    show_about()
