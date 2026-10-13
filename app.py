
import re
from urllib.parse import urlparse

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="HAN | Human Authenticity Network",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ HAN — Human Authenticity Network")
st.caption("Multimodal fraud screening: messages, URLs, images and QR codes")

st.warning(
    "Prototype only. Results are heuristic indicators, not proof of fraud "
    "or a guarantee that content is safe."
)

URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.I)

RISK_PATTERNS = {
    "OTP or credential request": (
        r"\b(share|send|reveal|enter|verify)\b.{0,35}"
        r"\b(otp|password|pin|cvv)\b"
    ),
    "Urgent account threat": (
        r"\b(account.{0,20}(blocked|suspended|closed)|"
        r"pay immediately|urgent payment|account will be blocked)\b"
    ),
    "Prize or reward scam language": (
        r"\b(you have won|claim your prize|lottery winner|free gift)\b"
    ),
    "Impersonation or secrecy indicators": (
        r"\b(don't tell anyone|keep this secret|"
        r"pretending to be|urgent transfer)\b"
    ),
}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly", "rb.gy"
}


def inspect_url(raw_url):
    url = raw_url.strip()
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url

    try:
        parsed = urlparse(url)
        host = parsed.hostname

        if not host:
            return 100, ["Invalid URL"]

        host = host.lower().rstrip(".")
        risk = 0
        reasons = []

        if parsed.scheme.lower() != "https":
            risk += 10
            reasons.append("URL does not use HTTPS")

        if "@" in parsed.netloc:
            risk += 30
            reasons.append("Contains @ in the authority section")

        if host in SHORTENERS:
            risk += 20
            reasons.append("Uses a URL-shortening service")

        if len(url) > 100:
            risk += 10
            reasons.append("Unusually long URL")

        if host.replace(".", "").isdigit():
            risk += 35
            reasons.append("Uses an IP address instead of a domain")

        if host.count(".") >= 4:
            risk += 10
            reasons.append("Unusually deep subdomain structure")

        if not reasons:
            reasons.append("No configured URL heuristic triggered")

        return min(risk, 100), reasons

    except (ValueError, UnicodeError):
        return 100, ["Could not parse URL"]


def scan_message(message):
    findings = []

    for description, pattern in RISK_PATTERNS.items():
        if re.search(pattern, message, re.I):
            findings.append(description)

    links = URL_PATTERN.findall(message)
    url_results = [inspect_url(link) for link in links]

    if links:
        findings.append(f"Contains {len(links)} URL(s)")

    url_risk = max((result[0] for result in url_results), default=0)
    score = min(100, len(findings) * 15 + url_risk)

    return score, findings, links, url_results


def show_result(score, reasons):
    if score >= 60:
        verdict = "REJECT"
    elif score >= 25:
        verdict = "VERIFY"
    else:
        verdict = "TRUST"

    st.subheader(f"Result: {verdict}")
    st.metric("Heuristic risk score", f"{score}/100")

    if reasons:
        st.write("**Signals detected:**")
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("No configured scam-language rules were triggered.")

    st.caption(
        "A low score does not guarantee safety. Verify suspicious requests "
        "through an independent, trusted channel."
    )


tab1, tab2, tab3 = st.tabs(
    ["Message / Email", "URL Scanner", "Image / QR Scanner"]
)

with tab1:
    st.subheader("Analyze a message or email")

    message = st.text_area(
        "Paste the suspicious content",
        placeholder="Your account will be blocked. Share your OTP...",
        height=180,
        key="message_input",
    )

    if st.button("Analyze Message"):
        if message.strip():
            score, findings, links, results = scan_message(message)
            show_result(score, findings)

            for link, (url_score, reasons) in zip(links, results):
                st.write(f"**URL:** `{link}`")
                st.write(f"URL risk: {url_score}/100")
                for reason in reasons:
                    st.write(f"- {reason}")
        else:
            st.info("Enter a message first.")

with tab2:
    st.subheader("Inspect a URL")

    url = st.text_input(
        "Enter a URL",
        placeholder="https://example.com",
        key="url_input",
    )

    if st.button("Analyze URL"):
        if url.strip():
            score, reasons = inspect_url(url)
            show_result(score, reasons)
            st.caption(
                "This checks URL structure only. It does not verify live "
                "domain reputation or open the website."
            )
        else:
            st.info("Enter a URL first.")

with tab3:
    st.subheader("Analyze an image or QR code")

    uploaded = st.file_uploader(
        "Upload a PNG, JPG or JPEG image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", width=450)

            extracted_text = pytesseract.image_to_string(image)
            st.text_area(
                "Text extracted by OCR",
                extracted_text or "No readable text found.",
                height=130,
            )

            detector = cv2.QRCodeDetector()
            qr_text, _, _ = detector.detectAndDecode(
                cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            )

            if qr_text:
                st.write("**QR content:**", qr_text)

            combined_text = extracted_text
            if qr_text:
                combined_text += "\n" + qr_text

            score, findings, links, results = scan_message(combined_text)

            if qr_text and not URL_PATTERN.search(qr_text):
                findings.append(
                    "QR contains non-URL data; verify its purpose"
                )

            show_result(score, findings)

            for link, (url_score, reasons) in zip(links, results):
                st.write(f"**Extracted URL:** `{link}`")
                st.write(f"URL risk: {url_score}/100")
                for reason in reasons:
                    st.write(f"- {reason}")

        except Exception as error:
            st.error(f"Could not process the image: {error}")

st.divider()
st.caption("HAN prototype | Explainable heuristic fraud screening")
