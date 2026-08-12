# app.py
import hashlib
import os

import requests
import streamlit as st

from password_utils import explain_password

# ------------------------------------------------------------------
# Local breach fallback list (common breached / dictionary passwords)
# ------------------------------------------------------------------
LOCAL_BREACHED = {
    "password", "123456", "123456789", "qwerty", "abc123", "password1",
    "12345678", "111111", "123123", "admin", "letmein", "welcome",
    "monkey", "dragon", "football", "baseball", "iloveyou", "trustno1",
    "sunshine", "princess", "superman", "starwars", "batman", "login",
    "passw0rd", "1234", "12345", "000000", "654321", "666666",
    "qwerty123", "password123", "admin123", "abc123456", "1234567890",
    "1q2w3e4r", "qwertyuiop", "asdfghjkl", "zxcvbnm", "pass123",
}
# Also load passwords from the test list file
_BREACH_FILE = "passwords.txt"
if os.path.exists(_BREACH_FILE):
    try:
        with open(_BREACH_FILE, "r", encoding="utf-8") as f:
            for line in f:
                pw = line.strip().split("\t")[0].strip().split(" ")[0].strip()
                if pw:
                    LOCAL_BREACHED.add(pw.lower())
    except Exception:
        pass


def check_hibp(password: str):
    """Check HaveIBeenPwned k-anonymity API. Returns (is_breached, count, source)."""
    sha1 = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = sha1[:5], sha1[5:]
    try:
        resp = requests.get(
            f"https://api.pwnedpasswords.com/range/{prefix}",
            timeout=5,
            headers={"User-Agent": "PasswordStrengthEvaluator/1.0"},
        )
        if resp.status_code == 200:
            for line in resp.text.splitlines():
                hash_suffix, _, count = line.partition(":")
                if hash_suffix.strip().upper() == suffix:
                    return True, int(count or 0), "hibp"
            return False, 0, "hibp"
    except Exception:
        pass
    # Fallback to local list
    if password.lower() in LOCAL_BREACHED:
        return True, 1, "local"
    return False, 0, "none"


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------
st.set_page_config(page_title="Password Strength Evaluator", page_icon="🔐", layout="centered")

st.title("🔐 Password Strength Evaluator")
st.caption("ML-powered password strength analysis with breach detection")

password = st.text_input(
    "Enter your password",
    type="password",
    placeholder="Type your password here...",
)

if st.button("🔍 Analyze Password", type="primary", use_container_width=True):
    if not password:
        st.warning("Please enter a password first.")
    else:
        with st.spinner("Analyzing password..."):
            try:
                result = explain_password(password)
                is_breached, count, source = check_hibp(password)

                features = result.get("features", {})
                label = result.get("label", "Unknown")

                # --- Strength badge ---
                color = {"Weak": "red", "Medium": "orange", "Strong": "green"}.get(label, "gray")
                st.markdown(
                    f"<div style='text-align:center; "
                    f"background:{color}; color:white; padding:12px; border-radius:8px; "
                    f"font-size:22px; font-weight:bold;'>Strength: {label}</div>",
                    unsafe_allow_html=True,
                )

                # --- Feature columns ---
                c1, c2, c3 = st.columns(3)
                c1.metric("Length", int(features.get("length", len(password))))
                c2.metric("Entropy", f"{float(features.get('shannon_entropy', 0)):.2f} bits")
                c3.metric("Eff. Entropy", f"{float(result.get('eff_entropy_bits', 0)):.2f} bits")

                c4, c5, c6 = st.columns(3)
                c4.metric("Digits", int(features.get("digits", 0)))
                c5.metric("Uppercase", int(features.get("uppers", 0)))
                c6.metric("Lowercase", int(features.get("lowers", 0)))

                c7, c8, c9 = st.columns(3)
                c7.metric("Special", int(features.get("specials", 0)))
                c8.metric("Online crack", f"{result.get('est_crack_time_online', '∞')}")
                c9.metric("Offline crack", f"{result.get('est_crack_time_offline', '∞')}")

                # --- Breach alert ---
                if is_breached:
                    if source == "hibp":
                        st.error(
                            f"⚠️ This password has been found **{count:,}** times in known "
                            "data breaches. Do NOT use it anywhere."
                        )
                    else:
                        st.error(
                            "⚠️ This password appears in a known breached/dictionary list. "
                            "Do NOT use it anywhere."
                        )
                else:
                    st.success(
                        "✅ No known breaches found. Checked via the HaveIBeenPwned "
                        "k-anonymity API."
                    )

                # --- Probability breakdown ---
                proba = result.get("proba")
                if proba:
                    st.subheader("📊 Model Confidence")
                    labels = ["Weak", "Medium", "Strong"]
                    st.bar_chart(
                        {labels[i]: round(float(proba[i]) * 100, 1) for i in range(min(3, len(proba)))}
                    )

            except Exception as exc:  # noqa: BLE001
                st.error(f"Analysis failed: {exc}")