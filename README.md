# SpamShield — Flask Spam Email Classifier

A beginner-friendly machine-learning web application that classifies an email as **spam** or **legitimate (ham)** from its subject and body.

## What this project does

1. Loads labeled email examples from `data/emails.csv`.
2. Normalizes the email text and replaces URLs, email addresses, and numbers with tokens.
3. Converts text into numeric features using **TF-IDF**, including single words and two-word phrases.
4. Trains a **Multinomial Naive Bayes** classifier, a common and efficient algorithm for text classification.
5. Shows the prediction and confidence in the Flask UI, or returns JSON through the API.

The model is trained when the app first needs it and cached in memory. For production, replace the small demo CSV with a larger representative dataset and evaluate on a held-out test set.

## Run on your computer

```bash
cd spam_email_classifier
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
flask --app app run --debug
```

Open **http://127.0.0.1:5000** in your browser.

## Test the API

```bash
curl http://127.0.0.1:5000/health
curl -X POST http://127.0.0.1:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"subject":"You won a prize","body":"Claim your cash reward now by clicking the link."}'
```

Example response:

```json
{"confidence": 93.0, "is_spam": true, "label": "spam", "sample_count": 50, "spam_probability": 93.0}
```

## Run automated tests

```bash
pip install pytest
pytest -q
```

## Project structure

```text
spam_email_classifier/
├── app.py                 # Flask routes and ML pipeline
├── data/emails.csv        # Labeled training examples
├── templates/index.html   # Web interface
├── static/style.css       # Styling
├── tests/test_app.py      # Health and prediction tests
└── requirements.txt       # Python dependencies
```

## Important note

This is an educational demonstration, not a mail-server filter. A real deployment should use thousands of current emails, avoid leaking private email content, tune the decision threshold, monitor false positives, and retrain when spam patterns change.
