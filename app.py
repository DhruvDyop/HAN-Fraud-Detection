"""
Drop-in Streamlit integration snippet for HAN-Fraud-Detection (app.py).

Add these cached model loaders and tabs to your existing app.py while keeping
your current scan_message() and inspect_url() functions intact.
"""

import os
import streamlit as st
from PIL import Image

from detectors.audio_detector import analyze_audio_bytes, load_audio_classifier
from detectors.deepfake_detector import analyze_video_bytes, load_deepfake_frame_classifier
from detectors.image_detector import analyze_image, load_image_classifier
from utils.trust_score import classify_risk_level, combine_multimodal_scores


@st.cache_resource(show_spinner="Loading AI image detector...")
def get_image_model():
    return load_image_classifier()


@st.cache_resource(show_spinner="Loading deepfake video detector...")
def get_video_model():
    return load_deepfake_frame_classifier()


@st.cache_resource(show_spinner="Loading AI voice detector...")
def get_audio_model():
    return load_audio_classifier()


# Example: update your st.tabs(...) call in app.py to include Video & Audio tabs
# tab_msg, tab_url, tab_img, tab_vid, tab_aud = st.tabs(
#     ["Message / Email", "URL Scanner", "Image / AI & QR", "Deepfake Video", "AI Voice / Audio"]
# )


def render_ai_image_section(pil_image: Image.Image, ocr_score: int = 0, qr_score: int = 0):
    """Call this inside your existing Image / QR tab after OCR & QR extraction."""
    st.subheader("AI-Generated Image & Metadata Forensics")
    use_local_ml = st.checkbox("Run local ML AI-image detector", value=True, key="img_ml_toggle")
    classifier = get_image_model() if use_local_ml else None

    result = analyze_image(pil_image, classifier=classifier)
    combined = combine_multimodal_scores(
        {"image": result["risk_score"], "message": ocr_score, "url": qr_score}
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("AI Synthetic Score", f"{result['risk_score']}/100")
    col2.metric("Combined Image Risk", f"{combined['risk_score']}/100")
    col3.metric("Verdict", combined["risk_level"])

    for sig in result["signals"]:
        st.write(f"- {sig}")
    if result["metadata"]:
        with st.expander("Extracted Image Metadata"):
            st.json(result["metadata"])


def render_deepfake_video_tab():
    """Render inside the new Deepfake Video tab."""
    st.subheader("Deepfake Video Screening")
    uploaded_video = st.file_uploader(
        "Upload a short video (.mp4, .mov, .avi, .webm)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
    )
    num_frames = st.slider("Frames to sample", min_value=6, max_value=24, value=12, step=2)

    if uploaded_video and st.button("Scan Video for Deepfakes", type="primary"):
        suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"
        classifier = get_video_model()
        with st.spinner("Sampling frames and analyzing faces..."):
            res = analyze_video_bytes(uploaded_video.read(), suffix=suffix, classifier=classifier, num_frames=num_frames)

        level, _ = classify_risk_level(res["risk_score"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Deepfake Risk", f"{res['risk_score']}/100")
        c2.metric("Verdict", level)
        c3.metric("Faces Detected", f"{res['faces_detected']}/{res['frames_analyzed']}")

        for sig in res["signals"]:
            st.write(f"- {sig}")

        if res.get("frame_timeline"):
            st.line_chart(
                {f"{pt['timestamp_sec']}s": pt["fake_probability"] * 100 for pt in res["frame_timeline"]}
            )

        if res.get("flagged_frames"):
            st.markdown("**Highest-Risk Sampled Frames**")
            cols = st.columns(len(res["flagged_frames"]))
            for col, frame in zip(cols, res["flagged_frames"]):
                col.image(
                    frame["thumbnail"],
                    caption=f"{frame['timestamp_sec']}s ({frame['fake_probability'] * 100:.1f}%)",
                    use_container_width=True,
                )


def render_audio_deepfake_tab():
    """Render inside the new AI Voice / Audio tab."""
    st.subheader("AI Voice & Audio Deepfake Screening")
    uploaded_audio = st.file_uploader(
        "Upload a voice note or call clip (.wav, .mp3, .m4a, .ogg)",
        type=["wav", "mp3", "m4a", "ogg"],
    )
    if uploaded_audio and st.button("Scan Audio for AI Voice", type="primary"):
        suffix = os.path.splitext(uploaded_audio.name)[1] or ".wav"
        classifier = get_audio_model()
        with st.spinner("Analyzing voice acoustics..."):
            res = analyze_audio_bytes(uploaded_audio.read(), suffix=suffix, classifier=classifier)

        level, _ = classify_risk_level(res["risk_score"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic Voice Risk", f"{res['risk_score']}/100")
        c2.metric("Verdict", level)
        c3.metric("Duration", f"{res['duration_sec']}s")

        for sig in res["signals"]:
            st.write(f"- {sig}")
