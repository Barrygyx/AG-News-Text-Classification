# AG News Text Classification

An end-to-end NLP classification system that combines pretrained sentence embeddings, classical machine learning, experiment tracking, API serving, containerization, and serverless deployment.

The application classifies news articles into four categories:

**World · Sports · Business · Sci/Tech**

**[Live Demo](YOUR_LAMBDA_FUNCTION_URL)**

---

## Project Overview

This project started as a model-comparison problem: given the title and description of a news article, predict one of four AG News categories.

Rather than directly fine-tuning a large language model, the project uses a pretrained sentence transformer as a feature extractor and compares several lightweight downstream classifiers. This separates language representation from classification and makes it possible to evaluate whether relatively simple models can perform well when provided with strong semantic embeddings.

The modeling pipeline was later extended into a complete inference system. Experiments are tracked with MLflow, the selected model is served through FastAPI, the application is packaged in Docker, and the production container is deployed on AWS Lambda behind a public Function URL.

The final deployed model is a **tuned Linear SVM operating on MiniLM sentence embeddings**, achieving **88.20% validation accuracy** and **87.85% Macro F1**.

---

## Modeling Pipeline

### 1. Data Preparation

The project uses an AG News subset with four target classes:

| Label | Category |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

Each observation contains a news **title** and **description**.

Instead of treating them as separate features, they are concatenated into a single text sequence:

```text
"title" + " " + "description"
```

This gives the sentence encoder both the short headline and the longer contextual description when constructing the document representation.

The labeled training split is used for model fitting and hyperparameter search, while a separate labeled validation split is reserved for final model comparison.

---

### 2. Sentence Representation with MiniLM

Raw text cannot be passed directly into the scikit-learn classifiers used in this project.

Each article is therefore encoded using the pretrained:

```text
all-MiniLM-L6-v2
```

sentence transformer.

MiniLM maps each article into a dense semantic embedding. The encoder itself is not fine-tuned during downstream classifier training; instead, it is used as a fixed feature extractor.

The modeling pipeline therefore becomes:

```text
News title + description
        ↓
all-MiniLM-L6-v2
        ↓
Dense sentence embedding
        ↓
Downstream classifier
        ↓
News category
```

This design makes the downstream experiments comparatively inexpensive because Logistic Regression, Linear SVM, and XGBoost are all trained on the same fixed embedding representation.

---

### 3. Baseline Model Comparison

Three downstream classifiers were initially evaluated:

#### Logistic Regression

A multinomial linear baseline was used to test how well the MiniLM embedding space could be separated using a relatively simple linear decision boundary.

#### Linear SVM

Linear SVM was included because high-dimensional embedding spaces are often well suited to margin-based linear classifiers.

#### XGBoost

XGBoost was included as a nonlinear tree-based alternative.

The baseline XGBoost configuration used:

```text
n_estimators = 200
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
objective = multi:softprob
```

All models were trained on exactly the same MiniLM representations to keep the comparison focused on the classifier rather than differences in text preprocessing.

---

## Evaluation Strategy

Models were compared using three main quantities:

- **Accuracy** — overall proportion of correctly classified articles
- **Macro F1** — F1 calculated independently for each class and averaged equally
- **Training time** — used to compare computational cost across models

Macro F1 was included because overall accuracy alone can hide differences in class-level performance.

The evaluation workflow also produces a classification report with class-specific precision, recall, and F1 scores for World, Sports, Business, and Sci/Tech.

### Baseline Results

| Model | Accuracy | Macro F1 | Training Time |
|---|---:|---:|---:|
| Logistic Regression | 87.40% | 87.05% | 0.54 s |
| Linear SVM | **87.60%** | **87.24%** | 0.65 s |
| XGBoost | 86.30% | 85.83% | 19.07 s |

The initial results showed that the linear models slightly outperformed XGBoost while requiring substantially less training time.

This suggested that the MiniLM embedding space was already structured enough that additional nonlinear model complexity was not necessary for this task.

---

## Hyperparameter Tuning

Because Logistic Regression and Linear SVM performed best in the initial comparison, both were selected for further tuning.

Hyperparameter optimization was implemented with **Optuna**.

The original training embeddings were divided into:

```text
80% optimization training split
20% tuning split
```

using a stratified split with a fixed random seed.

The external validation set was not used to select Optuna trials.

### Logistic Regression Search Space

Optuna explored:

```text
C: 1e-3 to 100, log scale
class_weight: None or balanced
```

Each configuration was evaluated using Macro F1 on the tuning split.

### Linear SVM Search Space

For Linear SVM, Optuna searched:

```text
C: 1e-3 to 100, log scale
class_weight: None or balanced
tol: 1e-5 to 1e-2, log scale
```

Each study contained **20 trials** and optimized Macro F1.

After tuning, the best model configuration was retrained using the complete training embedding set and evaluated once on the held-out validation split.

The selected Linear SVM configuration was approximately:

```text
C = 0.196
class_weight = balanced
tol = 1.03e-05
```

---

## Final Model Results

| Model | Validation Accuracy | Macro F1 |
|---|---:|---:|
| Tuned Logistic Regression | 87.90% | 87.61% |
| Tuned Linear SVM | **88.20%** | **87.85%** |

The tuned Linear SVM produced the strongest validation performance and was selected as the production classifier.

The final classifier is serialized with `joblib` and stored as:

```text
artifacts/tuned_linear_svm.pkl
```

An important result from the experiments is that greater model complexity did not automatically improve performance. XGBoost was substantially slower than the linear models and produced lower validation metrics, while tuning the Linear SVM produced the best overall result.

---

## Experiment Tracking with MLflow

Model comparison and tuning experiments are tracked using **MLflow**.

For each baseline run, the project records:

```text
model type
model hyperparameters
accuracy
Macro F1
training time
```

For example, the Linear SVM runs track:

```text
C
class_weight
tol
```

while XGBoost runs track:

```text
n_estimators
max_depth
learning_rate
```

The tuned Logistic Regression and Linear SVM experiments are also logged separately so that baseline and optimized models can be compared in the same experiment history.

MLflow uses a local SQLite tracking backend during development.

This separates experiment history from the training scripts and makes it easier to compare changes in model configuration without relying only on terminal output.

---

# Inference System

Training and inference are intentionally separated.

The training scripts are responsible for:

```text
dataset loading
→ MiniLM encoding
→ model training
→ evaluation
→ hyperparameter tuning
→ model serialization
```

The production application only performs the operations required for inference:

```text
user text
→ MiniLM embedding
→ saved Linear SVM
→ category prediction
```

This avoids retraining or loading the full AG News dataset inside the deployed application.

---

## How the API Works

The deployed service is built with **FastAPI**.

There are two main routes.

### `GET /`

The root route serves:

```text
frontend/index.html
```

which provides the browser interface used in the live demo.

### `POST /predict`

The frontend sends the user's article text to the prediction endpoint as JSON:

```json
{
  "text": "Apple reported stronger quarterly earnings..."
}
```

FastAPI validates the request using a Pydantic request model.

The request is then passed to:

```python
predict_news(text)
```

The inference sequence is:

```text
HTTP POST /predict
        ↓
FastAPI request validation
        ↓
predict_news()
        ↓
MiniLM encoder
        ↓
Sentence embedding
        ↓
Loaded tuned Linear SVM
        ↓
Integer class prediction
        ↓
Category mapping
        ↓
JSON response
```

A typical response is:

```json
{
  "text": "Apple reported stronger quarterly earnings...",
  "prediction": "Business"
}
```

The MiniLM encoder and trained SVM are loaded when the application starts rather than reloaded for every request.

---

# Deployment Architecture

The production system is currently deployed using AWS Lambda.

```text
Browser
   ↓
Lambda Function URL
   ↓
AWS Lambda
   ↓
Lambda Web Adapter
   ↓
Uvicorn
   ↓
FastAPI
   ↓
MiniLM + Linear SVM
   ↓
Prediction
```

The complete application is packaged as a Docker container.

The Docker image contains:

- FastAPI application
- frontend
- MiniLM model files
- tuned Linear SVM artifact
- Python dependencies
- AWS Lambda Web Adapter

The image is pushed to **Amazon ECR**, and the Lambda function runs that ECR image on demand.

---

## Why Lambda Web Adapter?

The FastAPI application is a normal HTTP web application running with Uvicorn.

Rather than rewriting the service into a Lambda-specific event handler, **AWS Lambda Web Adapter** translates incoming Lambda events into normal HTTP requests and forwards them to the web application running inside the container.

This allows essentially the same containerized FastAPI application to run locally or on AWS Lambda.

---

## Deployment Evolution

The first cloud version of the project was deployed using:

```text
Docker
→ Amazon ECR
→ ECS Express Mode
→ Fargate
→ Application Load Balancer
```

This successfully produced a public HTTPS application, but the infrastructure continued running even when the portfolio application had no traffic.

For a low-traffic demo, keeping a Fargate task and load balancer running continuously was unnecessary.

The application was therefore migrated to:

```text
Docker
→ Amazon ECR
→ AWS Lambda
→ Function URL
```

With the Lambda architecture, compute is invoked when the application receives requests instead of maintaining a continuously running task.

---

## Container Optimization

The initial Docker environment installed a CUDA-enabled PyTorch distribution even though the deployed model performs CPU inference.

This resulted in an image with approximately:

```text
3.4 GB content size
```

The deployment image was rebuilt with **CPU-only PyTorch**, reducing the image content size to approximately:

```text
610 MB
```

without changing model predictions.

This reduced unnecessary container dependencies and made the image more appropriate for serverless deployment.

---

# Repository Structure

```text
AG-News-Text-Classification/
│
├── artifacts/
│   ├── logistic_regression.pkl
│   ├── tuned_logistic_regression.pkl
│   └── tuned_linear_svm.pkl
│
├── frontend/
│   └── index.html
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   ├── models.py
│   ├── compare_models.py
│   ├── tune.py
│   ├── mlflow_compare.py
│   └── mlflow_tuned.py
│
├── api.py
├── data.py
├── features.py
├── predict.py
├── Dockerfile
├── requirements.txt
├── .dockerignore
└── .gitignore
```

### Main Components

`training/models.py`  
Defines Logistic Regression, Linear SVM, and XGBoost baseline models.

`training/compare_models.py`  
Creates MiniLM embeddings and compares model accuracy, Macro F1, class-level performance, and training time.

`training/tune.py`  
Uses Optuna to tune Logistic Regression and Linear SVM before training the final optimized models.

`training/mlflow_compare.py` and `training/mlflow_tuned.py`  
Record model parameters, metrics, and training time in MLflow.

`features.py`  
Loads MiniLM and converts text into sentence embeddings.

`predict.py`  
Loads the deployed SVM artifact and performs inference.

`api.py`  
Defines the FastAPI routes used by the frontend and prediction service.

`Dockerfile`  
Packages the web application and its inference dependencies for AWS Lambda.

---

# Reproducing the Application

The live application is available through the link at the top of this README.

To run the same container locally:

```bash
docker build -t ag-news-classifier .
docker run --rm -p 8000:8000 ag-news-classifier
```

Then open:

```text
http://localhost:8000
```

The same FastAPI application is used by both the local Docker container and the AWS Lambda deployment.
