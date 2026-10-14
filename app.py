
import ipaddress
import re
from urllib.parse import urlsplit

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# HAN: HUMAN AUTHENTICITY NETWORK
# --------------------------------------------------

st.set_page_config(
    page_title="HAN | Human Authenticity Network",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HAN — Human Authenticity Network")
st.caption(
    "Multimodal fraud screening: messages, URLs, images and QR codes"
)

st.warning(
    "Prototype only. Risk scores are heuristic indicators, not proof "
    "of fraud. A low score does not guarantee that content is safe."
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

URL_PATTERN = re.compile(
    r"""(?ix)
    \b(?:
        https?://[^\s<>"']+
        | www\.[^\s<>"']+
    )
    """
)

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "cutt.ly",
    "rb.gy",
    "rebrand.ly",
    "shorturl.at",
}

# Message indicators and their heuristic weights.
MESSAGE_PATTERNS = {
    "Credential or OTP request": (
        35,
        r"\b(?:share|send|reveal|enter|provide|confirm|"
        r"submit|type)\b.{0,50}\b(?:otp|one[- ]time password|"
        r"password|pin|cvv|card details|login credentials)\b",
    ),
    "Urgent account threat": (
        25,
        r"\b(?:account.{0,35}(?:blocked|suspended|locked|closed)|"
        r"will be (?:blocked|suspended|closed)|"
        r"immediate action required|act now or|"
        r"within \d+ hours?)\b",
    ),
    "Click or tap persuasion": (
        20,
        r"\b(?:click|tap|press|open|follow|visit)\b"
        r".{0,50}\b(?:here|this link|the link|below|"
        r"button|url|website|link)\b"
        r"|\b(?:click here|tap here|follow this link|"
        r"visit the link|open this link|click the link)\b",
    ),
    "Prize or reward claim": (
        25,
        r"\b(?:you have won|you've won|you.ve won|"
        r"claim your prize|lottery winner|free gift|"
        r"claim your reward|congratulations.{0,40}"
        r"(?:winner|prize|reward))\b",
    ),
    "Payment or fee pressure": (
        25,
        r"\b(?:pay|transfer|send)\b.{0,35}"
        r"\b(?:fee|payment|money|deposit|refund|fine|wallet)\b"
        r"|\b(?:pay immediately|urgent payment|payment failed)\b",
    ),
    "Delivery or parcel lure": (
        20,
        r"\b(?:parcel|package|shipment|delivery)\b"
        r".{0,45}\b(?:on hold|failed|held|address|fee|"
        r"reschedule|track|customs)\b",
    ),
    "Impersonation or secrecy indicators": (
        25,
        r"\b(?:don't tell anyone|do not tell anyone|"
        r"keep this secret|urgent transfer|"
        r"pretending to be|gift cards immediately)\b",
    ),
    "Security verification lure": (
        20,
        r"\b(?:verify|validate|restore|unlock|secure|"
        r"update|confirm)\b.{0,40}"
        r"\b(?:account|identity|bank|wallet|payment|"
        r"login|profile|details)\b",
    ),
    "Suspicious download request": (
        20,
        r"\b(?:download|install|run)\b.{0,45}"
        r"\b(?:attachment|file|app|application|update|software)\b",
    ),
}


# --------------------------------------------------
# URL ANALYSIS
# --------------------------------------------------

def inspect_url(raw_url):
    """Inspect URL syntax and configured risk indicators."""

    original = raw_url.strip()

    if not original:
        return 100, ["Empty URL"]

    if any(char.isspace() for char in original):
        return 100, ["URL contains whitespace or line breaks"]

    # Remove common punctuation surrounding links in messages.
    url = original.rstrip(".,!?;:)]}")

    risk = 0
    reasons = []

    # Detect malformed percent-encoding.
    if re.search(r"%(?![0-9a-fA-F]{2})", url):
        risk += 30
        reasons.append("Malformed percent-encoding")

    # Reject unsupported schemes such as javascript: and ftp:.
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url):
        if not re.match(r"^https?://", url, re.I):
            return 100, ["Unsupported or non-web URL scheme"]

    # Normalize URLs missing their scheme.
    if url.lower().startswith(("http://", "https://")):
        normalized_url = url
    elif url.lower().startswith("www."):
        normalized_url = "https://" + url
    else:
        normalized_url = "https://" + url

    try:
        parsed = urlsplit(normalized_url)

        if parsed.scheme.lower() not in {"http", "https"}:
            return 100, ["Unsupported URL scheme"]

        if not parsed.netloc or not parsed.hostname:
            return 100, ["Missing hostname or invalid URL syntax"]

        # Invalid port syntax raises ValueError.
        try:
            port = parsed.port
        except ValueError:
            return 100, ["Invalid port number in URL"]

        if parsed.netloc.endswith(":"):
            return 100, ["Port separator present but port is missing"]

        # Embedded credentials can disguise the destination.
        if parsed.username is not None or parsed.password is not None:
            risk += 35
            reasons.append(
                "URL contains embedded username or password"
            )

        if parsed.scheme.lower() == "http":
            risk += 10
            reasons.append("URL does not use HTTPS")

        host = parsed.hostname.lower().rstrip(".")

        # Validate an IP address or validate basic domain syntax.
        try:
            ipaddress.ip_address(host)
            risk += 30
            reasons.append("Uses an IP address instead of a domain")

        except ValueError:
            if len(host) > 253:
                return 100, ["Hostname is too long"]

            labels = host.split(".")

            invalid_labels = any(
                not label
                or len(label) > 63
                or label.startswith("-")
                or label.endswith("-")
                or not re.fullmatch(r"[a-z0-9-]+", label)
                for label in labels
            )

            if invalid_labels:
                return 100, ["Invalid hostname syntax"]

            if len(labels) < 2:
                risk += 10
                reasons.append(
                    "Hostname may be incomplete or local"
                )

        if host in SHORTENERS:
            risk += 20
            reasons.append("Uses a URL-shortening service")

        if host.count(".") >= 4:
            risk += 15
            reasons.append(
                "Unusually deep subdomain structure"
            )

        if len(normalized_url) > 150:
            risk += 10
            reasons.append("Unusually long URL")

        if port is not None and port not in {80, 443}:
            risk += 10
            reasons.append("Uses a non-standard web port")

        # Flag potentially deceptive URL keywords.
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

        if not reasons:
            reasons.append(
                "No configured URL heuristic triggered"
            )

        return min(risk, 100), reasons

    except (ValueError, UnicodeError):
        return 100, ["Could not parse URL"]


# --------------------------------------------------
# MESSAGE ANALYSIS
# --------------------------------------------------

def scan_message(message):
    """Detect suspicious message language and inspect its links."""

    findings = []
    message_risk = 0

    for description, (weight, pattern) in MESSAGE_PATTERNS.items():
        if re.search(pattern, message, re.I | re.S):
            findings.append(description)
            message_risk += weight

    # Extract and inspect links in the message.
    links = [
        match.rstrip(".,!?;:)]}")
        for match in URL_PATTERN.findall(message)
    ]

    url_results = [
        inspect_url(link)
        for link in links
    ]

    if links:
        findings.append(f"Contains {len(links)} URL(s)")

    # Increase risk when link-following language and a link coexist.
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

    # Separate message and URL risk for transparent reporting.
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


# --------------------------------------------------
# RESULTS DISPLAY
# --------------------------------------------------

def show_result(score, reasons, title="Analysis Result"):
    """Display a verdict and the reasons behind it."""

    st.subheader(title)

    if score >= 60:
        verdict = "REJECT / HIGH RISK"
        st.error(verdict)
    elif score >= 25:
        verdict = "VERIFY / CAUTION"
        st.warning(verdict)
    else:
        verdict = "LOW RISK INDICATORS"
        st.success(verdict)

    st.metric("Heuristic risk score", f"{score}/100")

    if reasons:
        st.write("**Signals detected:**")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("No configured risk indicators were triggered.")

    st.caption(
        "This score is not a probability of fraud. A low score "
        "does not prove that content or a link is safe."
    )


def show_url_details(links, results, label="URL"):
    """Display the analysis for each extracted URL."""

    for link, (url_score, reasons) in zip(links, results):
        st.divider()
        st.write(f"**{label}:** `{link}`")
        st.write(f"**URL risk:** {url_score}/100")

        for reason in reasons:
            st.write(f"- {reason}")


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "Message / Email",
        "URL Scanner",
        "Image / QR Scanner",
    ]
)


# --------------------------------------------------
# TAB 1: MESSAGE / EMAIL
# --------------------------------------------------

with tab1:
    st.subheader("Analyze a message or email")

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
            ) = scan_message(message)

            show_result(overall_score, findings, "Message Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Message language risk",
                    f"{message_score}/100",
                )

            with col2:
                st.metric(
                    "Highest URL risk",
                    f"{url_score}/100",
                )

            show_url_details(links, results)

        else:
            st.info("Enter a message first.")


# --------------------------------------------------
# TAB 2: URL SCANNER
# --------------------------------------------------

with tab2:
    st.subheader("Inspect a URL")

    st.caption(
        "Checks URL structure and configured indicators. "
        "It does not open the website or check live reputation."
    )

    url = st.text_input(
        "Enter a URL",
        placeholder="https://example.com",
        key="url_input",
    )

    if st.button("Analyze URL", key="analyze_url"):
        if url.strip():
            score, reasons = inspect_url(url)
            show_result(score, reasons, "URL Analysis")

        else:
            st.info("Enter a URL first.")


# --------------------------------------------------
# TAB 3: IMAGE / QR SCANNER
# --------------------------------------------------

with tab3:
    st.subheader("Analyze an image or QR code")

    uploaded = st.file_uploader(
        "Upload a PNG, JPG or JPEG image",
        type=["png", "jpg", "jpeg"],
        key="image_upload",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")

            st.image(
                image,
                caption="Uploaded image",
                width=450,
            )

            # OCR: extract text from the image.
            try:
                extracted_text = pytesseract.image_to_string(image)
            except Exception as error:
                extracted_text = ""
                st.warning(
                    f"OCR could not run. Check Tesseract installation: "
                    f"{error}"
                )

            st.text_area(
                "Text extracted by OCR",
                extracted_text or "No readable text found.",
                height=130,
                key="ocr_output",
            )

            # QR code decoding.
            detector = cv2.QRCodeDetector()

            image_array = cv2.cvtColor(
                np.array(image),
                cv2.COLOR_RGB2BGR,
            )

            qr_text, _, _ = detector.detectAndDecode(image_array)

            if qr_text:
                st.write("**QR content detected:**")
                st.code(qr_text)

            # Analyze both OCR text and QR content.
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
                ) = scan_message(combined_text)

                show_result(
                    overall_score,
                    findings,
                    "Image / QR Analysis",
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Extracted text risk",
                        f"{message_score}/100",
                    )

                with col2:
                    st.metric(
                        "Highest URL risk",
                        f"{url_score}/100",
                    )

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
                    "A clean scan does not establish that the image "
                    "or QR code is safe."
                )

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            cv2.error,
        ) as error:
            st.error(
                f"Could not process this image: {error}"
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "HAN prototype | Explainable heuristic fraud screening | "
    "No live domain reputation checks"
)
