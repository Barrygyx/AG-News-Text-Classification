from sklearn.linear_model import LogisticRegression
import joblib

from data import load_ag_news, combine_text
from features import load_encoder, create_embeddings


def train_model():

    # 1. Load dataset
    dataset = load_ag_news()

    # 2. Prepare training text and labels
    train_texts = combine_text(dataset["train"])
    y_train = dataset["train"]["label"]

    # 3. Load MiniLM encoder
    encoder = load_encoder()

    # 4. Convert text into embeddings
    print("Encoding training data...")
    X_train = create_embeddings(train_texts, encoder)

    # 5. Train Logistic Regression
    print("Training Logistic Regression...")

    classifier = LogisticRegression(
        max_iter=1000
    )

    classifier.fit(X_train, y_train)

    # 6. Save trained model
    joblib.dump(
        classifier,
        "logistic_regression.pkl"
    )

    print("Model saved successfully.")

    return classifier


if __name__ == "__main__":
    train_model()