import time
import numpy as np
import mlflow

from sklearn.metrics import accuracy_score, f1_score

from data import load_ag_news, combine_text
from features import load_encoder, create_embeddings
from models import get_models


# 1. Set up MLflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "AG News Model Comparison"
)


# 2. Load and prepare data

print("Loading dataset...")

dataset = load_ag_news()

train_texts = combine_text(
    dataset["train"]
)

y_train = np.array(
    dataset["train"]["label"]
)

val_texts = combine_text(
    dataset["validation"]
)

y_val = np.array(
    dataset["validation"]["label"]
)


# 3. Create MiniLM embeddings

print("Loading MiniLM encoder...")

encoder = load_encoder()

print("Encoding training data...")

X_train = create_embeddings(
    train_texts,
    encoder
)

print("Encoding validation data...")

X_val = create_embeddings(
    val_texts,
    encoder
)


# 4. Load candidate models

models = get_models()


# 5. Train and track each model

for name, model in models.items():

    print(f"\nTraining {name}...")

    with mlflow.start_run(
        run_name=name
    ):

        start_time = time.time()

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.time() - start_time
        )

        predictions = model.predict(
            X_val
        )

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )


        # 6. Log model information

        mlflow.log_param(
            "model_type",
            name
        )

        if name == "logistic_regression":

            mlflow.log_param(
                "C",
                model.C
            )

            mlflow.log_param(
                "max_iter",
                model.max_iter
            )


        elif name == "linear_svm":

            mlflow.log_param(
                "C",
                model.C
            )

            mlflow.log_param(
                "class_weight",
                model.class_weight
            )

            mlflow.log_param(
                "tol",
                model.tol
            )


        elif name == "xgboost":

            mlflow.log_param(
                "n_estimators",
                model.n_estimators
            )

            mlflow.log_param(
                "max_depth",
                model.max_depth
            )

            mlflow.log_param(
                "learning_rate",
                model.learning_rate
            )


        # 7. Log evaluation metrics

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "macro_f1",
            macro_f1
        )

        mlflow.log_metric(
            "training_time",
            training_time
        )


        # 8. Print results

        print(
            f"Accuracy: {accuracy:.4f}"
        )

        print(
            f"Macro F1: {macro_f1:.4f}"
        )

        print(
            f"Training time: "
            f"{training_time:.2f}s"
        )


print(
    "\nAll experiments have been logged to MLflow."
)