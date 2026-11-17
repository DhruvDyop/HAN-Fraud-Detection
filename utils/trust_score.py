"""Unified trust and fraud risk scoring utilities for HAN-Fraud-Detection."""

from __future__ import annotations

from typing import Any


def classify_risk_level(score: int | float) -> tuple[str, str]:
    """Map a 0-100 risk score to a human-readable verdict and color token."""
    clamped = max(0, min(100, int(round(score))))
    if clamped >= 70:
        return "High Risk", "red"
    if clamped >= 35:
        return "Medium Risk", "orange"
    return "Low Risk", "green"


def build_risk_report(
    modality: str,
    score: int | float,
    reasons: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized risk report dictionary across all HAN detectors."""
    clamped = max(0, min(100, int(round(score))))
    level, color = classify_risk_level(clamped)
    return {
        "modality": modality,
        "risk_score": clamped,
        "trust_score": 100 - clamped,
        "risk_level": level,
        "badge_color": color,
        "reasons": reasons or [],
        "details": details or {},
    }


def combine_multimodal_scores(
    modality_scores: dict[str, int | float],
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Aggregate scores from multiple modalities (e.g. OCR text + QR URL + AI image).

    Uses both weighted averaging and a high-signal floor (so a 95/100 phishing URL
    or 95/100 AI deepfake is never diluted below Medium/High risk).
    """
    if not modality_scores:
        return build_risk_report("combined", 0, reasons=["No modality signals provided."])

    default_weights = {
        "message": 0.30,
        "url": 0.35,
        "image": 0.35,
        "video": 0.50,
        "audio": 0.40,
    }
    active_weights = weights or default_weights

    weighted_sum = 0.0
    total_weight = 0.0
    for name, score in modality_scores.items():
        w = float(active_weights.get(name, 0.25))
        weighted_sum += max(0.0, min(100.0, float(score))) * w
        total_weight += w

    avg_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    max_score = max(float(s) for s in modality_scores.values())
    # Preserve strong single-modality red flags
    final_score = int(round(max(avg_score, 0.75 * max_score)))
    level, color = classify_risk_level(final_score)

    return {
        "risk_score": final_score,
        "trust_score": 100 - final_score,
        "risk_level": level,
        "badge_color": color,
        "breakdown": {k: int(round(v)) for k, v in modality_scores.items()},
    }
