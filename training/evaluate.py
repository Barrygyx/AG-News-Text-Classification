import joblib

from sklearn.metrics import accuracy_score, classification_report

from data import load_ag_news, combine_text, LABEL_NAMES
from features import load_encoder, create_embeddings


def evaluate_model():

    # 1. Load dataset
    dataset = load_ag_news()

    # 2. Prepare validation data
    val_texts = combine_text(dataset["validation"])
    y_val = dataset["validation"]["label"]

    # 3. Load MiniLM encoder
    encoder = load_encoder()

    # 4. Create embeddings
    print("Encoding validation data...")
    X_val = create_embeddings(val_texts, encoder)

    # 5. Load trained classifier
    classifier = joblib.load(
        "logistic_regression.pkl"
    )

    print("Model loaded successfully.")

    # 6. Make predictions
    predictions = classifier.predict(X_val)

    # 7. Evaluate performance
    accuracy = accuracy_score(
        y_val,
        predictions
    )

    print(f"\nValidation Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=list(LABEL_NAMES.values())
        )
    )


if __name__ == "__main__":
    evaluate_model()