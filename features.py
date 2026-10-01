from pathlib import Path

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "minilm_model"


def load_encoder():
    return SentenceTransformer(
        str(MODEL_PATH)
    )


def create_embeddings(texts, encoder):
    embeddings = encoder.encode(
        texts,
        show_progress_bar=False
    )

    return embeddings