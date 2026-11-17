"""AI voice and audio deepfake detection module using Wav2Vec2."""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np

DEFAULT_AUDIO_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
TARGET_SAMPLE_RATE = 16000


def load_audio_classifier(model_id: str = DEFAULT_AUDIO_MODEL):
    """Load a Hugging Face audio-classification pipeline (cache this in Streamlit)."""
    from transformers import pipeline

    return pipeline("audio-classification", model=model_id)


def _extract_audio_fake_score(predictions: list[dict[str, Any]]) -> tuple[float, str]:
    """Normalize audio classifier predictions into (fake_probability, top_label)."""
    if not predictions:
        return 0.0, "unknown"

    fake_tokens = {"fake", "spoof", "deepfake", "synthetic", "ai", "generated"}
    real_tokens = {"real", "human", "bona-fide", "bonafide", "authentic"}

    fake_prob = 0.0
    real_prob = 0.0
    top = max(predictions, key=lambda x: float(x.get("score", 0.0)))

    for item in predictions:
        label = str(item.get("label", "")).strip().lower()
        score = float(item.get("score", 0.0))
        if any(tok in label for tok in fake_tokens):
            fake_prob = max(fake_prob, score)
        elif any(tok in label for tok in real_tokens):
            real_prob = max(real_prob, score)

    if fake_prob == 0.0 and real_prob > 0.0:
        fake_prob = max(0.0, min(1.0, 1.0 - real_prob))

    return float(fake_prob), str(top.get("label", "unknown"))


def analyze_audio_bytes(
    audio_bytes: bytes,
    suffix: str = ".wav",
    classifier=None,
    max_seconds: int = 30,
) -> dict[str, Any]:
    """
    Analyze uploaded audio bytes (.wav, .mp3, .m4a, .ogg) for AI voice cloning.

    Resamples audio to 16 kHz mono and truncates to `max_seconds` for fast screening.
    """
    import librosa

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        waveform, sr = librosa.load(tmp_path, sr=TARGET_SAMPLE_RATE, mono=True, duration=max_seconds)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    duration_sec = round(float(len(waveform)) / TARGET_SAMPLE_RATE, 2) if len(waveform) > 0 else 0.0
    if len(waveform) == 0:
        return {
            "risk_score": 0,
            "fake_probability": 0.0,
            "label": "unknown",
            "duration_sec": 0.0,
            "signals": ["Uploaded audio file contains no decodable samples."],
        }

    fake_prob = 0.0
    top_label = "unknown"
    signals: list[str] = [f"Analyzed {duration_sec:.1f}s of 16 kHz mono audio."]

    if classifier is not None:
        preds = classifier({"array": np.asarray(waveform, dtype=np.float32), "sampling_rate": TARGET_SAMPLE_RATE})
        fake_prob, top_label = _extract_audio_fake_score(preds)
        signals.insert(0, f"Voice model '{top_label}' estimated {fake_prob * 100:.1f}% synthetic speech probability.")
    else:
        signals.append("Audio deepfake classifier not loaded; pass a cached pipeline to enable scoring.")

    risk_score = min(100, int(round(fake_prob * 100)))
    return {
        "risk_score": risk_score,
        "fake_probability": round(fake_prob, 4),
        "label": top_label,
        "duration_sec": duration_sec,
        "signals": signals,
    }
