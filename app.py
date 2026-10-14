
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import pytesseract
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ==================================================
# HAN — HUMAN AUTHENTICITY NETWORK
# ==================================================

st.set_page_config(
    page_title="HAN | Human Authenticity Network",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HAN — Human Authenticity Network")
st.caption("Multimodal fraud screening: messages, URLs, images and QR codes")

st.warning(
    "Prototype only. Scores are heuristic indicators, not proof of fraud. "
    "A URL not found in a threat database is not guaranteed to be safe."
)


# ==================================================
# CONFIGURATION
# ==================================================

URL_PATTERN = re.compile(
    r"""(?ix)\b(?:https?://[^\s<>"']+|www\.[^\s<>"']+)"""
)

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd",
    "cutt.ly", "rb.gy", "rebrand.ly", "shorturl.at",
}

# DEMO FIXTURE ONLY.
# This entry is included to test HAN's known-bad-domain handling.
# It is not an assertion that this domain is currently malicious.
KNOWN_BAD_TEST_DOMAINS = {
    "compromised-site.ru",
}

MESSAGE_PATTERNS = {
    "Credential or OTP request": (
        35,
        r"\b(?:share|send|reveal|enter|provide|confirm|submit|type)\b"
        r".{0,50}\b(?:otp|one[- ]time password|password|pin|cvv|"
        r"card details|login credentials)\b",
    ),
    "Urgent account threat": (
        25,
        r"\b(?:account.{0,35}(?:blocked|suspended|locked|closed)|"
        r"will be (?:blocked|suspended|closed)|"
        r"immediate action required|act now or|within \d+ hours?)\b",
    ),
    "Click or tap persuasion": (
        20,
        r"\b(?:click|tap|press|open|follow|visit)\b"
        r".{0,50}\b(?:here|this link|the link|below|button|url|"
        r"website|link)\b|"
        r"\b(?:click here|tap here|follow this link|visit the link|"
        r"open this link|click the link)\b",
    ),
    "Prize or reward claim": (
        25,
        r"\b(?:you have won|you've won|you.ve won|claim your prize|"
        r"lottery winner|free gift|claim your reward|"
        r"congratulations.{0,40}(?:winner|prize|reward))\b",
    ),
    "Payment or fee pressure": (
        25,
        r"\b(?:pay|transfer|send)\b.{0,35}"
        r"\b(?:fee|payment|money|deposit|refund|fine|wallet)\b|"
        r"\b(?:pay immediately|urgent payment|payment failed)\b",
    ),
    "Delivery or parcel lure": (
        20,
        r"\b(?:parcel|package|shipment|delivery)\b.{0,45}"
        r"\b(?:on hold|failed|held|address|fee|reschedule|track|customs)\b",
    ),
    "Impersonation or secrecy indicators": (
        25,
        r"\b(?:don't tell anyone|do not tell anyone|keep this secret|"
        r"urgent transfer|pretending to be|gift cards immediately)\b",
    ),
    "Security verification lure": (
        20,
        r"\b(?:verify|validate|restore|unlock|secure|update|confirm)\b"
        r".{0,40}\b(?:account|identity|bank|wallet|payment|login|"
        r"profile|details)\b",
    ),
    "Suspicious download request": (
        20,
        r"\b(?:download|install|run)\b.{0,45}"
        r"\b(?:attachment|file|app|application|update|software)\b",
    ),
}


# ==================================================
# URL HELPERS
# ==================================================

def normalize_url(raw_url):
    """Normalize a URL for inspection without opening it."""
    url = raw_url.strip().rstrip(".,!?;:)]}")

    if url.lower().startswith("www."):
        return "https://" + url

    if not re.match(r"^https?://", url, re.I):
        return "https://" + url

    return url


def get_validated_host(parsed):
    """Validate the hostname and return (host, error)."""
    host = parsed.hostname

    if not host:
        return None, "Missing hostname or invalid URL syntax"

    host = host.lower().rstrip(".")

    if len(host) > 253:
        return None, "Hostname is too long"

    try:
        ipaddress.ip_address(host)
        return host, None
    except ValueError:
        pass

    labels = host.split(".")

    if any(
        not label
        or len(label) > 63
        or label.startswith("-")
        or label.endswith("-")
        or not re.fullmatch(r"[a-z0-9-]+", label)
        for label in labels
    ):
        return None, "Invalid hostname syntax"

    if len(labels) < 2:
        return None, "Hostname may be incomplete or local"

    return host, None


# ==================================================
# GOOGLE SAFE BROWSING — OPTIONAL LIVE CHECK
# ==================================================

def check_google_safe_browsing(url):
    """
    Return (status, details).

    status is one of:
      malicious, clean_lookup, unavailable

    Requires GOOGLE_SAFE_BROWSING_API_KEY in Streamlit secrets.
    """
    try:
        api_key = st.secrets.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
    except Exception:
        api_key = ""

    if not api_key:
        return "unavailable", (
            "Google Safe Browsing is not configured. "
            "Only local heuristics and test fixtures were used."
        )

    endpoint = (
        "https://safebrowsing.googleapis.com/v4/"
        "threatMatches:find"
    )

    payload = {
        "client": {
            "clientId": "han-fraud-detection",
            "clientVersion": "1.0",
        },
        "threatInfo": {
            "threatTypes": [
                "MALWARE",
                "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE",
                "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }

    try:
        response = requests.post(
            endpoint,
            params={"key": api_key},
            json=payload,
            timeout=8,
        )

        if response.status_code != 200:
            return "unavailable", (
                f"Google Safe Browsing returned HTTP "
                f"{response.status_code}. Check the API key, "
                "API configuration, quota, and network."
            )

        data = response.json()

        if data.get("matches"):
            threat_types = sorted({
                item.get("threatType", "UNKNOWN")
                for item in data["matches"]
            })

            return "malicious", (
                "Google Safe Browsing reported a threat: "
                + ", ".join(threat_types)
            )

        return "clean_lookup", (
            "No match was returned by Google Safe Browsing. "
            "This does not prove the URL is safe."
        )

    except (requests.RequestException, ValueError) as error:
        return "unavailable", (
            f"Live reputation lookup failed: {error}"
        )


# ==================================================
# URL ANALYSIS
# ==================================================

def inspect_url(raw_url, use_live_lookup=False):
    """Return a heuristic score and an explainable list of findings."""
    original = raw_url.strip()

    if not original:
        return 100, ["Empty URL"]

    if any(char.isspace() for char in original):
        return 100, ["URL contains whitespace or line breaks"]

    url = original.rstrip(".,!?;:)]}")

    if re.search(r"%(?![0-9a-fA-F]{2})", url):
        return 100, ["Malformed percent-encoding"]

    # Do not silently accept non-web schemes.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url):
        if not re.match(r"^https?://", url, re.I):
            return 100, ["Unsupported or non-web URL scheme"]

    normalized_url = normalize_url(url)

    try:
        parsed = urlsplit(normalized_url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return 100, ["Unsupported URL scheme"]

        if not parsed.netloc or not parsed.hostname:
            return 100, ["Missing hostname or invalid URL syntax"]

        try:
            port = parsed.port
        except ValueError:
            return 100, ["Invalid port number in URL"]

        if parsed.netloc.endswith(":"):
            return 100, ["Port separator present but port is missing"]

        host, error = get_validated_host(parsed)

        if error:
            return 100, [error]

        # Check the local test fixture before calculating heuristics.
        if host in KNOWN_BAD_TEST_DOMAINS:
            return 100, [
                "Known-bad DEMO TEST FIXTURE matched",
                "This is a local test rule, not a live reputation result",
            ]

        risk = 0
        reasons = []

        if parsed.username is not None or parsed.password is not None:
            risk += 35
            reasons.append("URL contains embedded credentials")

        if parsed.scheme.lower() == "http":
            risk += 10
            reasons.append("URL does not use HTTPS")

        try:
            ipaddress.ip_address(host)
            risk += 30
            reasons.append("Uses an IP address instead of a domain")
        except ValueError:
            pass

        if host in SHORTENERS:
            risk += 20
            reasons.append("Uses a URL-shortening service")

        if host.count(".") >= 4:
            risk += 15
            reasons.append("Unusually deep subdomain structure")

        if len(normalized_url) > 150:
            risk += 10
            reasons.append("Unusually long URL")

        if port is not None and port not in {80, 443}:
            risk += 10
            reasons.append("Uses a non-standard web port")

        url_text = (
            host + parsed.path + "?" + parsed.query
        ).lower()

        if re.search(
            r"(?:login|verify|secure|account|wallet|password|"
            r"update|claim|bonus|prize|gift)",
            url_text,
        ):
            risk += 10
            reasons.append(
                "Contains account, verification, or reward keywords"
            )

        if "xn--" in host:
            risk += 20
            reasons.append(
                "Hostname contains internationalized/punycode labels"
            )

        # Optional live lookup. Never treat lookup failure as clean.
        if use_live_lookup:
            status, detail = check_google_safe_browsing(normalized_url)

            if status == "malicious":
                reasons.append(detail)
                risk = 100
            elif status == "clean_lookup":
                reasons.append(detail)
            else:
                reasons.append(detail)

        if not reasons:
            reasons.append("No configured URL heuristic triggered")

        return min(risk, 100), reasons

    except (ValueError, UnicodeError):
        return 100, ["Could not parse URL"]


# ==================================================
# MESSAGE ANALYSIS
# ==================================================

def scan_message(message, use_live_lookup=False):
    findings = []
    message_risk = 0

    for description, (weight, pattern) in MESSAGE_PATTERNS.items():
        if re.search(pattern, message, re.I | re.S):
            findings.append(description)
            message_risk += weight

    links = [
        match.rstrip(".,!?;:)]}")
        for match in URL_PATTERN.findall(message)
    ]

    url_results = [
        inspect_url(link, use_live_lookup=use_live_lookup)
        for link in links
    ]

    if links:
        findings.append(f"Contains {len(links)} URL(s)")

    click_language = re.search(
        r"\b(?:click|tap|open|follow|visit|press)\b"
        r".{0,50}\b(?:here|link|below|button|url|website)\b",
        message,
        re.I | re.S,
    )

    if links and click_language:
        findings.append(
            "Message urges the recipient to follow a link"
        )
        message_risk += 15

    max_url_risk = max(
        (score for score, _ in url_results),
        default=0,
    )

    message_score = min(message_risk, 100)
    overall_score = min(
        100,
        message_score + round(max_url_risk * 0.7),
    )

    return (
        overall_score,
        message_score,
        max_url_risk,
        findings,
        links,
        url_results,
    )


# ==================================================
# RESULTS DISPLAY
# ==================================================

def show_result(score, reasons, title="Analysis Result"):
    st.subheader(title)

    if score >= 60:
        st.error("HIGH RISK — DO NOT TRUST WITHOUT VERIFICATION")
    elif score >= 25:
        st.warning("CAUTION — VERIFY INDEPENDENTLY")
    else:
        st.success("LOW HEURISTIC RISK")

    st.metric("Heuristic risk score", f"{score}/100")

    if reasons:
        st.write("**Signals detected:**")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("No configured risk indicators were triggered.")

    st.caption(
        "Scores are not probabilities. A low score does not prove safety."
    )


def show_url_details(links, results, label="URL"):
    for link, (url_score, reasons) in zip(links, results):
        st.divider()
        st.write(f"**{label}:** `{link}`")
        st.write(f"**URL risk:** {url_score}/100")

        for reason in reasons:
            st.write(f"- {reason}")


# ==================================================
# DASHBOARD
# ==================================================

tab1, tab2, tab3 = st.tabs(
    ["Message / Email", "URL Scanner", "Image / QR Scanner"]
)

with tab1:
    st.subheader("Analyze a message or email")

    live_lookup = st.checkbox(
        "Use Google Safe Browsing for extracted URLs",
        value=False,
        help="Requires GOOGLE_SAFE_BROWSING_API_KEY in Streamlit Secrets.",
        key="message_live_lookup",
    )

    message = st.text_area(
        "Paste the suspicious content",
        placeholder=(
            "Your account will be blocked. "
            "Click here to verify your account..."
        ),
        height=180,
        key="message_input",
    )

    if st.button("Analyze Message", key="analyze_message"):
        if message.strip():
            (
                overall_score,
                message_score,
                url_score,
                findings,
                links,
                results,
            ) = scan_message(message, use_live_lookup=live_lookup)

            show_result(overall_score, findings, "Message Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Message language risk", f"{message_score}/100")

            with col2:
                st.metric("Highest URL risk", f"{url_score}/100")

            show_url_details(links, results)

        else:
            st.info("Enter a message first.")


with tab2:
    st.subheader("Inspect a URL")

    st.caption(
        "Checks URL syntax, local test fixtures, and configured heuristics. "
        "Optional reputation lookup requires an API key."
    )

    url = st.text_input(
        "Enter a URL",
        placeholder="https://example.com",
        key="url_input",
    )

    live_lookup = st.checkbox(
        "Check Google Safe Browsing",
        value=False,
        key="url_live_lookup",
    )

    if st.button("Analyze URL", key="analyze_url"):
        if url.strip():
            score, reasons = inspect_url(
                url,
                use_live_lookup=live_lookup,
            )
            show_result(score, reasons, "URL Analysis")
        else:
            st.info("Enter a URL first.")


with tab3:
    st.subheader("Analyze an image or QR code")

    image_live_lookup = st.checkbox(
        "Use Google Safe Browsing for extracted URLs",
        value=False,
        key="image_live_lookup",
    )

    uploaded = st.file_uploader(
        "Upload a PNG, JPG or JPEG image",
        type=["png", "jpg", "jpeg"],
        key="image_upload",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")

            st.image(image, caption="Uploaded image", width=450)

            try:
                extracted_text = pytesseract.image_to_string(image)
            except Exception as error:
                extracted_text = ""
                st.warning(f"OCR could not run: {error}")

            st.text_area(
                "Text extracted by OCR",
                extracted_text or "No readable text found.",
                height=130,
                key="ocr_output",
            )

            detector = cv2.QRCodeDetector()
            image_array = cv2.cvtColor(
                np.array(image),
                cv2.COLOR_RGB2BGR,
            )
            qr_text, _, _ = detector.detectAndDecode(image_array)

            if qr_text:
                st.write("**QR content detected:**")
                st.code(qr_text)

            combined_text = extracted_text or ""

            if qr_text:
                combined_text += "\n" + qr_text

            if combined_text.strip():
                (
                    overall_score,
                    message_score,
                    url_score,
                    findings,
                    links,
                    results,
                ) = scan_message(
                    combined_text,
                    use_live_lookup=image_live_lookup,
                )

                if qr_text and not URL_PATTERN.search(qr_text):
                    findings.append(
                        "QR contains non-URL data; verify its purpose"
                    )

                show_result(
                    overall_score,
                    findings,
                    "Image / QR Analysis",
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Extracted text risk", f"{message_score}/100")

                with col2:
                    st.metric("Highest URL risk", f"{url_score}/100")

                show_url_details(
                    links,
                    results,
                    label="Extracted URL",
                )

            else:
                st.info(
                    "No readable text or QR content was detected."
                )
                st.caption(
                    "A clean scan does not establish that an image or QR "
                    "code is safe."
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            cv2.error,
        ) as error:
            st.error(f"Could not process this image: {error}")


st.divider()
st.caption(
    "HAN prototype | Explainable heuristic screening | "
    "Optional Google Safe Browsing lookup"
)
