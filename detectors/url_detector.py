"""Modular URL phishing, lookalike domain, and Safe Browsing detector for HAN-Fraud-Detection."""

from __future__ import annotations

import ipaddress
import os
from typing import Any
from urllib.parse import urlparse

import requests
import tldextract

URL_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "cutt.ly",
    "rb.gy",
    "rebrand.ly",
    "shorturl.at",
}

SUSPICIOUS_URL_KEYWORDS = (
    "login",
    "verify",
    "secure",
    "account",
    "wallet",
    "password",
    "update",
    "claim",
    "bonus",
    "prize",
    "gift",
)

TEST_BLOCKLIST = {"compromised-site.ru"}


def check_google_safe_browsing(url: str, api_key: str | None = None) -> dict[str, Any]:
    """Query Google Safe Browsing Lookup API v4 if an API key is configured."""
    key = api_key or os.getenv("GOOGLE_SAFE_BROWSING_API_KEY")
    if not key:
        return {"checked": False, "flagged": False, "threats": []}

    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={key}"
    payload = {
        "client": {"clientId": "han-fraud-detection", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=10)
        if not resp.ok:
            return {"checked": False, "flagged": False, "threats": [], "error": f"HTTP {resp.status_code}"}
        matches = resp.json().get("matches", [])
        threats = [m.get("threatType", "UNKNOWN") for m in matches]
        return {"checked": True, "flagged": bool(threats), "threats": threats}
    except requests.RequestException as exc:
        return {"checked": False, "flagged": False, "threats": [], "error": str(exc)}


def inspect_url(url: str, safe_browsing_api_key: str | None = None) -> dict[str, Any]:
    """Analyze a URL for phishing heuristics, structural anomalies, and Safe Browsing matches."""
    raw = (url or "").strip()
    if not raw:
        return {"url": raw, "risk_score": 0, "signals": ["Empty URL provided."], "safe_browsing": {}}

    normalized = raw if "://" in raw else f"https://{raw}"
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()

    if not host:
        return {"url": normalized, "risk_score": 40, "signals": ["Malformed URL without a valid host."], "safe_browsing": {}}

    score = 0
    signals: list[str] = []
    extracted = tldextract.extract(host)
    registered_domain = extracted.top_domain_under_public_suffix or host

    if registered_domain in TEST_BLOCKLIST or host in TEST_BLOCKLIST:
        score += 80
        signals.append(f"Domain '{host}' matched local test blocklist (+80).")

    if parsed.username or parsed.password:
        score += 35
        signals.append("URL contains embedded credentials (@) (+35).")

    try:
        ipaddress.ip_address(host)
        score += 30
        signals.append("Host is a raw IP address instead of a domain (+30).")
    except ValueError:
        pass

    if registered_domain in URL_SHORTENERS or host in URL_SHORTENERS:
        score += 20
        signals.append(f"Uses URL shortener '{registered_domain}' (+20).")

    if "xn--" in host:
        score += 20
        signals.append("Contains Punycode ('xn--'), possible IDN homograph (+20).")

    if host.count(".") >= 4:
        score += 15
        signals.append("Deeply nested subdomains (>= 4 dots) (+15).")

    if parsed.scheme.lower() != "https":
        score += 10
        signals.append("Non-HTTPS scheme (+10).")

    if len(normalized) > 150:
        score += 10
        signals.append("Unusually long URL (> 150 characters) (+10).")

    if parsed.port and parsed.port not in (80, 443):
        score += 10
        signals.append(f"Non-standard port {parsed.port} (+10).")

    lower_url = normalized.lower()
    hit_keywords = [kw for kw in SUSPICIOUS_URL_KEYWORDS if kw in lower_url]
    if hit_keywords:
        score += 10
        signals.append(f"Suspicious phishing keywords ({', '.join(hit_keywords[:4])}) (+10).")

    sb_result = check_google_safe_browsing(normalized, api_key=safe_browsing_api_key)
    if sb_result.get("flagged"):
        score = max(score + 50, 90)
        signals.insert(0, f"Flagged by Google Safe Browsing: {', '.join(sb_result['threats'])}.")

    return {
        "url": normalized,
        "host": host,
        "registered_domain": registered_domain,
        "risk_score": min(100, score),
        "signals": signals or ["No obvious URL phishing indicators detected."],
        "safe_browsing": sb_result,
    }
