FROM public.ecr.aws/docker/library/python:3.11-slim

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.1 /lambda-adapter /opt/extensions/lambda-adapter

ENV PORT=8000

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/app/minilm_model')"

RUN chmod -R a+rX /app/minilm_model

COPY api.py .
COPY predict.py .
COPY features.py .
COPY data.py .
COPY tuned_linear_svm.pkl .

COPY frontend ./frontend

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]