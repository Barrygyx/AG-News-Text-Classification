import time
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, classification_report

from data import load_ag_news, combine_text, LABEL_NAMES
from features import load_encoder, create_embeddings
from models import get_models


def compare_models():

    # 1. Load dataset

    print("Loading dataset...")

    dataset = load_ag_news()


    # 2. Prepare training data

    print("Preparing training data...")

    train_texts = combine_text(
        dataset["train"]
    )

    y_train = np.array(dataset["train"]["label"])


    # 3. Prepare validation data

    print("Preparing validation data...")

    val_texts = combine_text(
        dataset["validation"]
    )

    y_val = np.array(dataset["validation"]["label"])


    # 4. Load MiniLM encoder

    print("Loading MiniLM encoder...")

    encoder = load_encoder()


    # 5. Create MiniLM embeddings

    print("\nEncoding training data...")

    X_train = create_embeddings(
        train_texts,
        encoder
    )


    print("\nEncoding validation data...")

    X_val = create_embeddings(
        val_texts,
        encoder
    )


    # 6. Load models

    models = get_models()

    results = []


    # 7. Train and evaluate each model

    for name, model in models.items():

        print("\n")
        print(f"Training {name}...")


        # Training

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )


        # Prediction

        predictions = model.predict(
            X_val
        )


        # Accuracy

        accuracy = accuracy_score(
            y_val,
            predictions
        )


        # Macro F1

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )


        # Print main results

        print(
            f"\nAccuracy: {accuracy:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        print(
            f"Training time: {training_time:.2f} seconds"
        )


        # Classification Report

        print(
            "\nClassification Report:"
        )

        print(
            classification_report(
                y_val,
                predictions,
                labels=[0, 1, 2, 3],
                target_names=list(
                    LABEL_NAMES.values()
                )
            )
        )


        # Save result

        results.append(
            {
                "model": name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "training_time": training_time
            }
        )


    # 8. Final comparison

    print("\n")
    print("FINAL MODEL COMPARISON")

    for result in results:

        print(
            f"{result['model']:22} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f} | "
            f"Time: {result['training_time']:.2f}s"
        )


# Run program

if __name__ == "__main__":
    compare_models()