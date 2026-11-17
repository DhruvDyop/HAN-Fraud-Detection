"""Deepfake video detection module using OpenCV frame sampling + Hugging Face ViT."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import cv2
import numpy as np
from PIL import Image

from detectors.image_detector import _extract_ai_score_from_predictions

DEFAULT_VIDEO_FRAME_MODEL = "prithivMLmods/Deep-Fake-Detector-v2-Model"


def load_deepfake_frame_classifier(model_id: str = DEFAULT_VIDEO_FRAME_MODEL):
    """Load a ViT image classifier for face/frame deepfake scoring (cache in Streamlit)."""
    from transformers import pipeline

    return pipeline("image-classification", model=model_id)


def _detect_largest_face(frame_rgb: np.ndarray, face_cascade: cv2.CascadeClassifier) -> tuple[Image.Image, bool]:
    """Crop the largest detected face with padding, or return the full frame if none found."""
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48, 48))

    if len(faces) == 0:
        return Image.fromarray(frame_rgb), False

    x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)
    h_img, w_img = frame_rgb.shape[:2]

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w_img, x + w + pad_w)
    y2 = min(h_img, y + h + pad_h)

    return Image.fromarray(frame_rgb[y1:y2, x1:x2]), True


def sample_video_frames(video_path: str, num_frames: int = 12) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Uniformly sample frames from a video file and return RGB arrays + timestamps."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video appears to have zero readable frames.")

    indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
    sampled: list[dict[str, Any]] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        sampled.append(
            {
                "frame_index": int(idx),
                "timestamp_sec": round(float(idx) / fps, 2),
                "image_rgb": frame_rgb,
            }
        )

    cap.release()
    return sampled, {"total_frames": total_frames, "fps": round(fps, 2), "duration_sec": round(duration_sec, 2)}


def analyze_video_bytes(
    video_bytes: bytes,
    suffix: str = ".mp4",
    classifier=None,
    num_frames: int = 12,
    max_flagged_thumbs: int = 4,
) -> dict[str, Any]:
    """
    Analyze uploaded video bytes for deepfake manipulation.

    Uniformly samples `num_frames` frames, crops the primary face in each frame
    via OpenCV Haar cascades, and aggregates per-frame deepfake probabilities.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        frames, meta = sample_video_frames(tmp_path, num_frames=num_frames)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not frames:
        return {
            "risk_score": 0,
            "fake_probability": 0.0,
            "faces_detected": 0,
            "frames_analyzed": 0,
            "video_meta": meta,
            "flagged_frames": [],
            "signals": ["No frames could be decoded from the uploaded video."],
        }

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    frame_results: list[dict[str, Any]] = []
    faces_detected = 0

    for item in frames:
        crop_pil, has_face = _detect_largest_face(item["image_rgb"], face_cascade)
        if has_face:
            faces_detected += 1

        fake_prob = 0.0
        label = "unknown"
        if classifier is not None:
            preds = classifier(crop_pil)
            fake_prob, label = _extract_ai_score_from_predictions(preds)

        frame_results.append(
            {
                "timestamp_sec": item["timestamp_sec"],
                "frame_index": item["frame_index"],
                "face_detected": has_face,
                "fake_probability": round(fake_prob, 4),
                "label": label,
                "thumbnail": crop_pil.resize((224, 224)),
            }
        )

    probs = [r["fake_probability"] for r in frame_results]
    avg_prob = float(np.mean(probs))
    peak_prob = float(np.max(probs))
    # Weight both average and peak frame suspicion so brief face-swaps still surface
    combined_prob = 0.6 * avg_prob + 0.4 * peak_prob
    risk_score = min(100, int(round(combined_prob * 100))) if classifier is not None else 0

    flagged = sorted(frame_results, key=lambda r: r["fake_probability"], reverse=True)[:max_flagged_thumbs]
    signals: list[str] = [
        f"Analyzed {len(frame_results)} sampled frames ({faces_detected} with detected faces).",
        f"Average fake probability: {avg_prob * 100:.1f}% | Peak frame: {peak_prob * 100:.1f}%.",
    ]
    if faces_detected == 0:
        signals.append("No frontal face detected; scored full frames instead.")
    if classifier is None:
        signals.append("Deepfake classifier not loaded; pass a cached pipeline to enable scoring.")

    return {
        "risk_score": risk_score,
        "fake_probability": round(combined_prob, 4),
        "avg_fake_probability": round(avg_prob, 4),
        "peak_fake_probability": round(peak_prob, 4),
        "faces_detected": faces_detected,
        "frames_analyzed": len(frame_results),
        "video_meta": meta,
        "frame_timeline": [
            {"timestamp_sec": r["timestamp_sec"], "fake_probability": r["fake_probability"]}
            for r in frame_results
        ],
        "flagged_frames": flagged,
        "signals": signals,
    }
