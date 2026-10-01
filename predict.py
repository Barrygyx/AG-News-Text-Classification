from pathlib import Path

import joblib

from data import LABEL_NAMES
from features import load_encoder, create_embeddings


BASE_DIR = Path(__file__).resolve().parent

encoder = load_encoder()

classifier = joblib.load(
    BASE_DIR / "tuned_linear_svm.pkl"
)


def predict_news(text):
    embedding = create_embeddings(
        [text],
        encoder
    )

    prediction = classifier.predict(
        embedding
    )[0]

    category = LABEL_NAMES[
        int(prediction)
    ]

    return category


if __name__ == "__main__":
    text = input(
        "Enter a news article: "
    )

    category = predict_news(
        text
    )

    print(
        "Predicted category:",
        category
    )