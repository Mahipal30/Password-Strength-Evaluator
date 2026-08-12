# 🔐 Password Strength Evaluator

An ML-powered password strength analyzer with breach detection, built with Streamlit.
Link:
Streamlit live url:https://mahipal30-password-strength-evaluator-app-pymsaz.streamlit.app/
## Features

- **Machine Learning classification** — A calibrated Random Forest classifier predicts password strength as **Weak / Medium / Strong**.
- **Rich feature extraction** — 25 features including Shannon entropy, character-class bits, LZ complexity, QWERTY/ASCII sequence patterns, n-gram surprisal, and common password patterns.
- **Breach detection** — Checks passwords against the [HaveIBeenPwned](https://haveibeenpwned.com/) k-anonymity API (a local fallback list is used if offline).
- **Crack-time estimates** — Estimates online and offline brute-force cracking times based on effective entropy.
- **Explainable output** — Shows feature breakdown, entropy, and model confidence probabilities.

## Project Structure

```
├── app.py                 # Streamlit UI + breach check
├── password_utils.py      # Feature extraction, prediction, explanation
├── trainmodel.py          # Trains & saves the ML model artifacts
├── data.csv               # Training dataset (password, strength)
├── passwords.txt          # Test cases
├── models/                # Saved model, scaler, ngram, meta (generated)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── ngram.pkl
│   └── meta.json
├── Templates/index.html   # (legacy) Flask-style frontend
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Train the model (optional)

The `models/` directory already contains trained artifacts. To retrain:

```bash
python trainmodel.py
```

This reads `data.csv`, extracts features, trains a calibrated Random Forest, and saves the artifacts to `models/`.

## Run the app

```bash
streamlit run app.py
```

Then open the printed local URL (default `http://localhost:8501`).

## Test cases

`passwords.txt` contains sample passwords with expected strength labels and reasoning — useful for manual testing.

## How it works

1. `password_utils.extract_features()` converts a password into a 25-dimension numeric vector.
2. A `StandardScaler` normalizes the features.
3. A calibrated `RandomForestClassifier` predicts the strength class.
4. `explain_password()` also computes effective entropy and crack-time estimates.
5. The Streamlit app displays results and checks the password against breach databases.

## CI / GitHub Actions

This repository includes a CI workflow (`.github/workflows/ci.yml`) that runs on every push/PR to `main`. It:

- Tests on Python 3.11 and 3.12
- Installs dependencies from `requirements.txt`
- Verifies the ML model artifacts load and predict correctly
- Verifies the local breach fallback list
- Runs a Streamlit app smoke test (checks the server responds with HTTP 200)

You can view workflow runs under the **Actions** tab of the repository.
