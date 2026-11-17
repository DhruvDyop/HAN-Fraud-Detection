"""HAN – Human Authenticity Network: Multimodal Fraud & Deepfake Screening App."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

from detectors.audio_detector import analyze_audio_bytes, load_audio_classifier
from detectors.deepfake_detector import analyze_video_bytes, load_deepfake_frame_classifier
from detectors.image_detector import analyze_image, load_image_classifier
from detectors.text_detector import scan_message
from detectors.url_detector import inspect_url
from utils.trust_score import build_risk_report, classify_risk_level, combine_multimodal_scores

st.set_page_config(
    page_title="HAN – Human Authenticity Network",
    page_icon="🛡️",
    layout="wide",
)


def _get_secret(key: str) -> str | None:
    """Safely read an optional secret from st.secrets or environment variables."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key)


@st.cache_resource(show_spinner="Loading AI image detector...")
def get_image_model():
    return load_image_classifier()


@st.cache_resource(show_spinner="Loading deepfake video detector...")
def get_video_model():
    return load_deepfake_frame_classifier()


@st.cache_resource(show_spinner="Loading AI voice detector...")
def get_audio_model():
    return load_audio_classifier()


def _extract_qr_payloads(pil_image: Image.Image) -> list[str]:
    """Decode any QR codes present in an uploaded image using OpenCV."""
    bgr = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    detector = cv2.QRCodeDetector()
    payloads: list[str] = []
    ok, decoded_info, _, _ = detector.detectAndDecodeMulti(bgr)
    if ok and decoded_info:
        payloads.extend(p.strip() for p in decoded_info if p and p.strip())
    else:
        single, _, _ = detector.detectAndDecode(bgr)
        if single and single.strip():
            payloads.append(single.strip())
    return list(dict.fromkeys(payloads))


def _extract_ocr_text(pil_image: Image.Image) -> tuple[str, str | None]:
    """Extract text from an image via Tesseract OCR, returning (text, error_msg)."""
    try:
        text = pytesseract.image_to_string(pil_image.convert("RGB")).strip()
        return text, None
    except Exception as exc:
        return "", f"OCR skipped ({exc})"


def render_message_tab(sb_key: str | None):
    st.subheader("Message / Email Phishing Scanner")
    msg = st.text_area(
        "Paste suspicious SMS, WhatsApp message, or email text",
        height=160,
        placeholder="e.g. Urgent: Your bank account is suspended. Verify OTP at http://bit.ly/...",
    )
    if st.button("Scan Message", type="primary", key="btn_scan_msg"):
        if not msg.strip():
            st.warning("Please enter message text to scan.")
            return
        res = scan_message(msg, safe_browsing_api_key=sb_key)
        report = build_risk_report("message", res["risk_score"], reasons=res["signals"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{report['risk_score']}/100")
        c2.metric("Trust Score", f"{report['trust_score']}/100")
        c3.metric("Verdict", report["risk_level"])

        for sig in res["signals"]:
            st.write(f"- {sig}")

        if res["url_reports"]:
            st.markdown("**Extracted URLs**")
            for u_rep in res["url_reports"]:
                u_lvl, _ = classify_risk_level(u_rep["risk_score"])
                st.write(f"- `{u_rep['url']}` — **{u_rep['risk_score']}/100 ({u_lvl})**")


def render_url_tab(sb_key: str | None):
    st.subheader("URL & Phishing Link Inspector")
    url_input = st.text_input("Enter URL to inspect", placeholder="https://example.com/login")
    if st.button("Inspect URL", type="primary", key="btn_scan_url"):
        if not url_input.strip():
            st.warning("Please enter a URL to inspect.")
            return
        res = inspect_url(url_input, safe_browsing_api_key=sb_key)
        report = build_risk_report("url", res["risk_score"], reasons=res["signals"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{report['risk_score']}/100")
        c2.metric("Trust Score", f"{report['trust_score']}/100")
        c3.metric("Verdict", report["risk_level"])

        for sig in res["signals"]:
            st.write(f"- {sig}")


def render_image_tab(sb_key: str | None, hf_token: str | None):
    st.subheader("AI Image, Metadata, OCR & QR Scanner")
    uploaded_img = st.file_uploader(
        "Upload an image (.png, .jpg, .jpeg, .webp)",
        type=["png", "jpg", "jpeg", "webp"],
        key="img_uploader",
    )
    use_local_ml = st.checkbox("Run local ML AI-image detector", value=True, key="img_ml_toggle")

    if uploaded_img:
        pil_image = Image.open(uploaded_img)
        st.image(pil_image, caption="Uploaded Image", width=360)

        if st.button("Analyze Image", type="primary", key="btn_scan_img"):
            classifier = get_image_model() if use_local_ml else None
            img_res = analyze_image(pil_image, classifier=classifier, hf_api_token=hf_token)

            ocr_text, ocr_err = _extract_ocr_text(pil_image)
            ocr_res = scan_message(ocr_text, safe_browsing_api_key=sb_key) if ocr_text else {"risk_score": 0, "signals": []}

            qr_payloads = _extract_qr_payloads(pil_image)
            qr_reports = [inspect_url(p, safe_browsing_api_key=sb_key) for p in qr_payloads]
            qr_score = max((r["risk_score"] for r in qr_reports), default=0)

            combined = combine_multimodal_scores(
                {"image": img_res["risk_score"], "message": ocr_res["risk_score"], "url": qr_score}
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("AI Synthetic Score", f"{img_res['risk_score']}/100")
            c2.metric("Combined Image Risk", f"{combined['risk_score']}/100")
            c3.metric("Verdict", combined["risk_level"])

            st.markdown("**AI & Metadata Signals**")
            for sig in img_res["signals"]:
                st.write(f"- {sig}")

            if ocr_err:
                st.info(ocr_err)
            elif ocr_text:
                with st.expander(f"OCR Text (Risk: {ocr_res['risk_score']}/100)"):
                    st.text(ocr_text)

            if qr_reports:
                st.markdown("**Decoded QR Payloads**")
                for qr in qr_reports:
                    st.write(f"- `{qr['url']}` — **{qr['risk_score']}/100**")

            if img_res["metadata"]:
                with st.expander("Extracted Image Metadata"):
                    st.json(img_res["metadata"])


def render_deepfake_video_tab():
    st.subheader("Deepfake Video Screening")
    uploaded_video = st.file_uploader(
        "Upload a short video (.mp4, .mov, .avi, .webm)",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="vid_uploader",
    )
    num_frames = st.slider("Frames to sample", min_value=6, max_value=24, value=12, step=2)

    if uploaded_video and st.button("Scan Video for Deepfakes", type="primary", key="btn_scan_vid"):
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
            st.line_chart({f"{pt['timestamp_sec']}s": pt["fake_probability"] * 100 for pt in res["frame_timeline"]})

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
    st.subheader("AI Voice & Audio Deepfake Screening")
    uploaded_audio = st.file_uploader(
        "Upload a voice note or call clip (.wav, .mp3, .m4a, .ogg)",
        type=["wav", "mp3", "m4a", "ogg"],
        key="aud_uploader",
    )
    if uploaded_audio and st.button("Scan Audio for AI Voice", type="primary", key="btn_scan_aud"):
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


def main():
    st.title("🛡️ HAN – Human Authenticity Network")
    st.caption("Multimodal fraud screening: phishing messages, URLs, AI images & QR codes, deepfake videos, and cloned voices.")

    sb_key = _get_secret("GOOGLE_SAFE_BROWSING_API_KEY")
    hf_token = _get_secret("HF_TOKEN")

    tab_msg, tab_url, tab_img, tab_vid, tab_aud = st.tabs(
        ["Message / Email", "URL Scanner", "Image / AI & QR", "Deepfake Video", "AI Voice / Audio"]
    )
    with tab_msg:
        render_message_tab(sb_key)
    with tab_url:
        render_url_tab(sb_key)
    with tab_img:
        render_image_tab(sb_key, hf_token)
    with tab_vid:
        render_deepfake_video_tab()
    with tab_aud:
        render_audio_deepfake_tab()


if __name__ == "__main__":
    main()
