"""AI-generated image and metadata forensics detector for HAN-Fraud-Detection."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from PIL import ExifTags, Image

DEFAULT_IMAGE_MODEL = "prithivMLmods/deepfake-detector-model-v1"
FALLBACK_IMAGE_MODEL = "Bombek1/ai-image-detector-siglip-dinov2"

AI_GENERATOR_KEYWORDS = (
    "midjourney",
    "dall-e",
    "dalle",
    "stable diffusion",
    "automatic1111",
    "comfyui",
    "invokeai",
    "firefly",
    "flux",
    "leonardo",
    "ideogram",
    "bing image creator",
    "ai generated",
    "synthetic",
)


def load_image_classifier(model_id: str = DEFAULT_IMAGE_MODEL):
    """Load a Hugging Face image-classification pipeline (cache this in Streamlit)."""
    from transformers import pipeline

    return pipeline("image-classification", model=model_id)


def inspect_image_metadata(pil_image: Image.Image) -> dict[str, Any]:
    """Inspect EXIF and PNG text chunks for camera provenance or AI generator tags."""
    flags: list[str] = []
    metadata_summary: dict[str, str] = {}

    # Check PNG / info text chunks (often used by Stable Diffusion / ComfyUI)
    for key, value in (pil_image.info or {}).items():
        val_str = str(value)
        if key.lower() in {"parameters", "prompt", "workflow", "software", "comment", "description"}:
            metadata_summary[str(key)] = val_str[:200]
        lower_val = val_str.lower()
        for kw in AI_GENERATOR_KEYWORDS:
            if kw in lower_val:
                flags.append(f"Generator signature '{kw}' found in image metadata ({key}).")
                break

    # Check EXIF tags
    exif_raw = pil_image.getexif()
    if not exif_raw:
        flags.append("No EXIF camera metadata present (common in AI-generated or stripped web images).")
    else:
        exif_named = {ExifTags.TAGS.get(k, str(k)): str(v) for k, v in exif_raw.items()}
        has_camera = any(tag in exif_named for tag in ("Make", "Model", "LensModel"))
        if not has_camera:
            flags.append("EXIF metadata lacks camera Make/Model tags.")
        for tag in ("Software", "Artist", "ImageDescription", "Make", "Model"):
            if tag in exif_named:
                metadata_summary[tag] = exif_named[tag][:120]
                if any(kw in exif_named[tag].lower() for kw in AI_GENERATOR_KEYWORDS):
                    flags.append(f"AI tool reference found in EXIF {tag}: {exif_named[tag][:80]}")

    return {
        "flags": list(dict.fromkeys(flags)),
        "metadata": metadata_summary,
    }


def _extract_ai_score_from_predictions(predictions: list[dict[str, Any]]) -> tuple[float, str]:
    """Normalize Hugging Face classifier outputs into (ai_probability, top_label)."""
    if not predictions:
        return 0.0, "unknown"

    ai_labels = {"fake", "deepfake", "ai", "artificial", "synthetic", "generated", "ai_generated"}
    real_labels = {"real", "human", "realism", "authentic", "natural"}

    ai_prob = 0.0
    real_prob = 0.0
    top = max(predictions, key=lambda x: float(x.get("score", 0.0)))

    for item in predictions:
        label = str(item.get("label", "")).strip().lower()
        score = float(item.get("score", 0.0))
        if any(tok in label for tok in ai_labels):
            ai_prob = max(ai_prob, score)
        elif any(tok in label for tok in real_labels):
            real_prob = max(real_prob, score)

    if ai_prob == 0.0 and real_prob > 0.0:
        ai_prob = max(0.0, min(1.0, 1.0 - real_prob))

    return float(ai_prob), str(top.get("label", "unknown"))


def _frequency_artifact_hint(pil_image: Image.Image) -> float:
    """Lightweight FFT high-frequency energy ratio as a secondary heuristic (0.0 to 1.0)."""
    gray = np.asarray(pil_image.convert("L").resize((256, 256)), dtype=np.float32)
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(f_shift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.ogrid[:h, :w]
    dist = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
    outer_ring = magnitude[dist > 64].mean()
    total_mean = magnitude.mean() + 1e-6
    ratio = float(outer_ring / total_mean)
    return float(max(0.0, min(1.0, (ratio - 0.55) / 0.45)))


def analyze_image(
    pil_image: Image.Image,
    classifier=None,
    hf_api_token: str | None = None,
    model_id: str = DEFAULT_IMAGE_MODEL,
) -> dict[str, Any]:
    """
    Analyze an image for AI synthesis / deepfake manipulation.

    Pass a cached Hugging Face `pipeline("image-classification", ...)` via `classifier`
    for fast local inference, or provide `hf_api_token` (or set `HF_TOKEN`) to query
    the Hugging Face Inference API without loading local PyTorch weights.
    """
    rgb_image = pil_image.convert("RGB")
    meta = inspect_image_metadata(pil_image)
    signals: list[str] = list(meta["flags"])

    ai_prob = 0.0
    top_label = "unknown"
    backend = "metadata-only"

    if classifier is not None:
        predictions = classifier(rgb_image)
        ai_prob, top_label = _extract_ai_score_from_predictions(predictions)
        backend = f"local:{model_id}"
    else:
        token = hf_api_token or os.getenv("HF_TOKEN")
        if token:
            import io
            import requests

            buf = io.BytesIO()
            rgb_image.save(buf, format="PNG")
            resp = requests.post(
                f"https://api-inference.huggingface.co/models/{model_id}",
                headers={"Authorization": f"Bearer {token}"},
                data=buf.getvalue(),
                timeout=30,
            )
            if resp.ok and isinstance(resp.json(), list):
                ai_prob, top_label = _extract_ai_score_from_predictions(resp.json())
                backend = f"hf-api:{model_id}"

    # Combine model probability with metadata & spectral hints into a 0-100 risk score
    meta_bonus = 15 if any("Generator signature" in f or "AI tool" in f for f in meta["flags"]) else 0
    if any("No EXIF" in f for f in meta["flags"]):
        meta_bonus += 5

    freq_hint = _frequency_artifact_hint(rgb_image)
    if backend == "metadata-only":
        risk_score = min(100, int(round(meta_bonus + freq_hint * 40)))
        signals.append("ML classifier not loaded; score reflects metadata and spectral heuristics only.")
    else:
        risk_score = min(100, int(round(ai_prob * 85 + meta_bonus)))
        signals.insert(0, f"Model '{top_label}' estimated {ai_prob * 100:.1f}% synthetic probability.")

    return {
        "ai_probability": round(ai_prob, 4),
        "label": top_label,
        "risk_score": risk_score,
        "metadata_flags": meta["flags"],
        "metadata": meta["metadata"],
        "signals": signals,
        "backend": backend,
    }
