import time
import joblib
import numpy as np
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data import load_ag_news, combine_text
from features import load_encoder, create_embeddings


# 1. Load and prepare data

print("Loading dataset...")
dataset = load_ag_news()

train_texts = combine_text(dataset["train"])
y_train = np.array(dataset["train"]["label"])

final_val_texts = combine_text(dataset["validation"])
y_final_val = np.array(dataset["validation"]["label"])


# 2. Create MiniLM embeddings

print("Loading MiniLM encoder...")
encoder = load_encoder()

print("Encoding training data...")
X_train_all = create_embeddings(train_texts, encoder)

print("Encoding final validation data...")
X_final_val = create_embeddings(final_val_texts, encoder)


# 3. Split training data for tuning

X_train, X_tune, y_train_split, y_tune = train_test_split(
    X_train_all,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train
)


# 4. Tune Logistic Regression

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

    model.fit(X_train, y_train_split)

    predictions = model.predict(X_tune)

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

print(
    "Best tuning Macro F1:",
    round(logistic_study.best_value, 4)
)


# 5. Tune Linear SVM

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

    model.fit(X_train, y_train_split)

    predictions = model.predict(X_tune)

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

print(
    "Best tuning Macro F1:",
    round(svm_study.best_value, 4)
)


# 6. Train tuned Logistic Regression on full training data

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

print("\nFinal Logistic Regression")
print(f"Accuracy: {logistic_accuracy:.4f}")
print(f"Macro F1: {logistic_f1:.4f}")
print(f"Training time: {logistic_time:.2f}s")

joblib.dump(
    best_logistic,
    "tuned_logistic_regression.pkl"
)


# 7. Train tuned Linear SVM on full training data

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

print("\nFinal Linear SVM")
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Macro F1: {svm_f1:.4f}")
print(f"Training time: {svm_time:.2f}s")

joblib.dump(
    best_svm,
    "tuned_linear_svm.pkl"
)


# 8. Compare tuned models

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