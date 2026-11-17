"""Modular text, phishing message, and AI-scam-text detector for HAN-Fraud-Detection."""

from __future__ import annotations

import re
from typing import Any

from detectors.url_detector import inspect_url

URL_REGEX = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)

# Weighted regex categories matching HAN-Fraud-Detection's message heuristics
MESSAGE_PATTERNS: list[tuple[str, int, re.Pattern[str]]] = [
    (
        "Credential or OTP request",
        35,
        re.compile(r"\b(otp|one[- ]time password|pin|cvv|passcode|password|seed phrase|recovery phrase)\b", re.I),
    ),
    (
        "Urgent account threat",
        25,
        re.compile(r"\b(suspended|locked|deactivated|unauthorized|within 24 hours|immediately|urgent action)\b", re.I),
    ),
    (
        "Prize or reward claim",
        25,
        re.compile(r"\b(won|winner|lottery|jackpot|claim your|congratulations|gift card|cash prize)\b", re.I),
    ),
    (
        "Payment or fee pressure",
        25,
        re.compile(r"\b(wire transfer|upi|crypto|bitcoin|usdt|processing fee|customs fee|overdue invoice|pay now)\b", re.I),
    ),
    (
        "Impersonation or secrecy indicator",
        25,
        re.compile(r"\b(kindly|do not share|confidential|ceo|hr department|bank manager|tax authority|kyc)\b", re.I),
    ),
    (
        "Click or tap persuasion",
        20,
        re.compile(r"\b(click here|tap below|verify now|login now|update your account|confirm identity)\b", re.I),
    ),
    (
        "Delivery or parcel lure",
        20,
        re.compile(r"\b(parcel|package|delivery failed|reschedule delivery|shipment|courier)\b", re.I),
    ),
    (
        "Security verification lure",
        20,
        re.compile(r"\b(security alert|unusual sign[- ]in|verify your identity|account verification)\b", re.I),
    ),
    (
        "Suspicious download request",
        20,
        re.compile(r"\b(download attachment|open pdf|enable macros|install apk|\.exe|\.apk)\b", re.I),
    ),
]


def load_ai_text_classifier(model_id: str = "roberta-base-openai-detector"):
    """Optional Hugging Face text-classification pipeline to flag AI-generated scam copy."""
    from transformers import pipeline

    return pipeline("text-classification", model=model_id, truncation=True, max_length=512)


def extract_urls(text: str) -> list[str]:
    """Extract unique URLs from message text, normalizing bare www. prefixes."""
    seen: list[str] = []
    for raw in URL_REGEX.findall(text or ""):
        cleaned = raw.rstrip(".,;:!?)")
        if cleaned.lower().startswith("www."):
            cleaned = f"https://{cleaned}"
        if cleaned not in seen:
            seen.append(cleaned)
    return seen


def scan_message(
    text: str,
    safe_browsing_api_key: str | None = None,
    ai_text_classifier=None,
) -> dict[str, Any]:
    """
    Screen a message or email body for phishing/scam heuristics, embedded URLs,
    and optional AI-generated text signals.
    """
    content = (text or "").strip()
    if not content:
        return {
            "risk_score": 0,
            "matched_categories": [],
            "extracted_urls": [],
            "url_reports": [],
            "ai_text_probability": 0.0,
            "signals": ["No message text provided."],
        }

    score = 0
    matched_categories: list[dict[str, Any]] = []
    signals: list[str] = []

    for label, weight, pattern in MESSAGE_PATTERNS:
        matches = pattern.findall(content)
        if matches:
            score += weight
            matched_categories.append({"category": label, "weight": weight, "matches": list(dict.fromkeys(matches))})
            signals.append(f"{label} (+{weight})")

    extracted_urls = extract_urls(content)
    url_reports = [inspect_url(u, safe_browsing_api_key=safe_browsing_api_key) for u in extracted_urls]
    max_url_score = max((r["risk_score"] for r in url_reports), default=0)

    if url_reports:
        signals.append(f"Found {len(url_reports)} embedded URL(s); highest URL risk is {max_url_score}/100.")
        score = max(score, int(round(0.6 * score + 0.5 * max_url_score)))

    ai_prob = 0.0
    if ai_text_classifier is not None and len(content) >= 40:
        preds = ai_text_classifier(content[:1500])
        if preds:
            top = preds[0]
            lbl = str(top.get("label", "")).lower()
            prob = float(top.get("score", 0.0))
            ai_prob = prob if any(k in lbl for k in ("fake", "ai", "generated")) else (1.0 - prob)
            if ai_prob >= 0.7:
                signals.append(f"Message style resembles AI-generated text ({ai_prob * 100:.1f}% confidence).")

    return {
        "risk_score": min(100, score),
        "matched_categories": matched_categories,
        "extracted_urls": extracted_urls,
        "url_reports": url_reports,
        "ai_text_probability": round(ai_prob, 4),
        "signals": signals or ["No high-risk phishing patterns detected."],
    }
