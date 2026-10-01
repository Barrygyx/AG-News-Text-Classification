import time
import numpy as np
import optuna
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data import load_ag_news, combine_text
from features import load_encoder, create_embeddings


# 1. Set up MLflow

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

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

final_val_texts = combine_text(
    dataset["validation"]
)

y_final_val = np.array(
    dataset["validation"]["label"]
)


# 3. Create MiniLM embeddings

print("Loading MiniLM encoder...")

encoder = load_encoder()

print("Encoding training data...")

X_train_all = create_embeddings(
    train_texts,
    encoder
)

print("Encoding validation data...")

X_final_val = create_embeddings(
    final_val_texts,
    encoder
)


# 4. Split training data for Optuna

X_train, X_tune, y_train_split, y_tune = train_test_split(
    X_train_all,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


# 5. Tune Logistic Regression

def objective_logistic(trial):

    C = trial.suggest_float(
        "C",
        1e-3,
        100,
        log=True
    )

    class_weight = trial.suggest_categorical(
        "class_weight",
        [None, "balanced"]
    )

    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=2000
    )

    model.fit(
        X_train,
        y_train_split
    )

    predictions = model.predict(
        X_tune
    )

    macro_f1 = f1_score(
        y_tune,
        predictions,
        average="macro"
    )

    return macro_f1


print("\nTuning Logistic Regression...")

logistic_study = optuna.create_study(
    direction="maximize"
)

logistic_study.optimize(
    objective_logistic,
    n_trials=20
)

print("\nBest Logistic Regression parameters:")
print(logistic_study.best_params)


# 6. Train final tuned Logistic Regression

best_logistic = LogisticRegression(
    **logistic_study.best_params,
    max_iter=2000
)

start_time = time.time()

best_logistic.fit(
    X_train_all,
    y_train
)

logistic_time = time.time() - start_time

logistic_predictions = best_logistic.predict(
    X_final_val
)

logistic_accuracy = accuracy_score(
    y_final_val,
    logistic_predictions
)

logistic_f1 = f1_score(
    y_final_val,
    logistic_predictions,
    average="macro"
)


# 7. Log tuned Logistic Regression to MLflow

with mlflow.start_run(
    run_name="tuned_logistic_regression"
):

    mlflow.log_param(
        "model_type",
        "logistic_regression"
    )

    mlflow.log_param(
        "stage",
        "tuned"
    )

    for param_name, param_value in logistic_study.best_params.items():

        mlflow.log_param(
            param_name,
            param_value
        )

    mlflow.log_param(
        "max_iter",
        2000
    )

    mlflow.log_metric(
        "accuracy",
        logistic_accuracy
    )

    mlflow.log_metric(
        "macro_f1",
        logistic_f1
    )

    mlflow.log_metric(
        "training_time",
        logistic_time
    )


print("\nTuned Logistic Regression")
print(f"Accuracy: {logistic_accuracy:.4f}")
print(f"Macro F1: {logistic_f1:.4f}")
print(f"Training time: {logistic_time:.2f}s")


# 8. Tune Linear SVM

def objective_svm(trial):

    C = trial.suggest_float(
        "C",
        1e-3,
        100,
        log=True
    )

    class_weight = trial.suggest_categorical(
        "class_weight",
        [None, "balanced"]
    )

    tolerance = trial.suggest_float(
        "tol",
        1e-5,
        1e-2,
        log=True
    )

    model = LinearSVC(
        C=C,
        class_weight=class_weight,
        tol=tolerance
    )

    model.fit(
        X_train,
        y_train_split
    )

    predictions = model.predict(
        X_tune
    )

    macro_f1 = f1_score(
        y_tune,
        predictions,
        average="macro"
    )

    return macro_f1


print("\nTuning Linear SVM...")

svm_study = optuna.create_study(
    direction="maximize"
)

svm_study.optimize(
    objective_svm,
    n_trials=20
)

print("\nBest Linear SVM parameters:")
print(svm_study.best_params)


# 9. Train final tuned Linear SVM

best_svm = LinearSVC(
    **svm_study.best_params
)

start_time = time.time()

best_svm.fit(
    X_train_all,
    y_train
)

svm_time = time.time() - start_time

svm_predictions = best_svm.predict(
    X_final_val
)

svm_accuracy = accuracy_score(
    y_final_val,
    svm_predictions
)

svm_f1 = f1_score(
    y_final_val,
    svm_predictions,
    average="macro"
)


# 10. Log tuned Linear SVM to MLflow

with mlflow.start_run(
    run_name="tuned_linear_svm"
):

    mlflow.log_param(
        "model_type",
        "linear_svm"
    )

    mlflow.log_param(
        "stage",
        "tuned"
    )

    for param_name, param_value in svm_study.best_params.items():

        mlflow.log_param(
            param_name,
            param_value
        )

    mlflow.log_metric(
        "accuracy",
        svm_accuracy
    )

    mlflow.log_metric(
        "macro_f1",
        svm_f1
    )

    mlflow.log_metric(
        "training_time",
        svm_time
    )


print("\nTuned Linear SVM")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Macro F1: {svm_f1:.4f}")
print(f"Training time: {svm_time:.2f}s")


# 11. Print final comparison

print("\nTuned Model Comparison")

print(
    f"{'Logistic Regression':22} | "
    f"Accuracy: {logistic_accuracy:.4f} | "
    f"Macro F1: {logistic_f1:.4f}"
)

print(
    f"{'Linear SVM':22} | "
    f"Accuracy: {svm_accuracy:.4f} | "
    f"Macro F1: {svm_f1:.4f}"
)

print("\nTuned models have been logged to MLflow.")