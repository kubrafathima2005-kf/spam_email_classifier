from flask import Flask, render_template, request
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


app = Flask(__name__)

MODEL_FILE = "spam_model.pkl"


def load_dataset():
    """
    Finds a CSV file inside the data folder.
    """

    data_folder = "data"

    if not os.path.exists(data_folder):
        raise FileNotFoundError("data folder not found.")

    csv_files = [
        file for file in os.listdir(data_folder)
        if file.lower().endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            "No CSV file found inside the data folder."
        )

    file_path = os.path.join(data_folder, csv_files[0])

    print("Using dataset:", file_path)

    df = pd.read_csv(file_path)

    print("Dataset columns:", df.columns.tolist())

    return df


def prepare_data(df):
    """
    Automatically detects common column names
    for spam/ham and email/message text.
    """

    # Find label column
    label_column = None

    possible_label_columns = [
        "label",
        "Category",
        "category",
        "spam",
        "target",
        "class"
    ]

    for column in possible_label_columns:
        if column in df.columns:
            label_column = column
            break

    # Find text column
    text_column = None

    possible_text_columns = [
        "text",
        "message",
        "Message",
        "email",
        "Email",
        "body",
        "content"
    ]

    for column in possible_text_columns:
        if column in df.columns:
            text_column = column
            break

    if label_column is None:
        raise ValueError(
            "Could not find spam/ham label column. "
            "Expected something like: label, category, spam or class."
        )

    if text_column is None:
        raise ValueError(
            "Could not find email text column. "
            "Expected something like: text, message, email or body."
        )

    print("Label column:", label_column)
    print("Text column:", text_column)

    df = df[[text_column, label_column]].dropna()

    X = df[text_column].astype(str)
    y = df[label_column].astype(str)

    return X, y


def train_model():

    print("\nLoading dataset...")

    df = load_dataset()

    X, y = prepare_data(df)

    print("Training model...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            MultinomialNB()
        )
    ])

    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)

    print("Model trained successfully!")
    print("Model saved as:", MODEL_FILE)

    return model


def get_model():

    if os.path.exists(MODEL_FILE):
        print("Loading existing model...")
        return joblib.load(MODEL_FILE)

    return train_model()


model = get_model()


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    probability = None
    email_text = ""

    if request.method == "POST":

        email_text = request.form.get("email", "").strip()

        if not email_text:
            prediction = "Please enter an email message."

        else:

            prediction_result = model.predict([email_text])[0]

            # Get probability if available
            try:
                probabilities = model.predict_proba([email_text])[0]
                probability = round(max(probabilities) * 100, 2)
            except Exception:
                probability = None

            label = str(prediction_result).lower()

            if label in ["spam", "1", "true"]:
                prediction = "SPAM"

            else:
                prediction = "NOT SPAM"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        email_text=email_text
    )


if __name__ == "__main__":
    app.run(debug=True)