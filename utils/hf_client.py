"""
Shared Hugging Face Inference API client with retry logic and error handling.
"""

import time
import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient


@st.cache_resource
def get_client() -> InferenceClient:
    """Create and cache the HF InferenceClient using Streamlit secrets."""
    token = st.secrets.get("HF_TOKEN")
    if not token:
        st.error(
            "**HF_TOKEN not found.** Please add your Hugging Face token to "
            "`.streamlit/secrets.toml` as `HF_TOKEN = \"hf_...\"`."
        )
        st.stop()
    return InferenceClient(token=token)


def safe_summarize(
    text: str,
    model: str,
    retries: int = 3,
) -> str | None:
    """
    Summarize text via HF Inference API with retry on 503 (model loading)
    and 429 (rate limit).
    """
    client = get_client()
    for attempt in range(retries):
        try:
            result = client.summarization(
                text,
                model=model,
            )
            return result.summary_text
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg:
                with st.spinner(
                    f"⏳ Model is loading... attempt {attempt + 1}/{retries} (this can take up to 60s)"
                ):
                    time.sleep(20)
            elif "429" in error_msg:
                with st.spinner("⏳ Rate limit reached. Waiting..."):
                    time.sleep(10)
            else:
                st.error(f"API Error: {error_msg}")
                return None
    st.error("❌ Model unavailable after multiple attempts. Please try again later.")
    return None


def get_embeddings(
    texts: list[str],
    model: str,
    retries: int = 3,
) -> np.ndarray | None:
    """
    Get sentence embeddings via HF Inference API (feature-extraction).
    Returns a numpy array of shape (n_texts, embedding_dim).
    """
    client = get_client()
    for attempt in range(retries):
        try:
            results = client.feature_extraction(
                texts,
                model=model,
            )
            embeddings = np.array(results)
            # Handle different response shapes
            if embeddings.ndim == 3:
                # (n_texts, n_tokens, dim) → mean pooling
                embeddings = embeddings.mean(axis=1)
            return embeddings
        except Exception as e:
            error_msg = str(e)
            if "503" in error_msg:
                with st.spinner(
                    f"⏳ Embedding model loading... attempt {attempt + 1}/{retries}"
                ):
                    time.sleep(20)
            elif "429" in error_msg:
                with st.spinner("⏳ Rate limit reached. Waiting..."):
                    time.sleep(10)
            else:
                st.error(f"API Error: {error_msg}")
                return None
    st.error("❌ Embedding model unavailable. Please try again later.")
    return None


def compute_similarity(
    emb1: np.ndarray,
    emb2: np.ndarray,
) -> dict[str, float]:
    """
    Compute multiple similarity/distance metrics between two embedding vectors.
    Returns dict with cosine similarity, euclidean distance, and dot product.
    """
    # Flatten to 1D if needed
    e1 = emb1.flatten()
    e2 = emb2.flatten()

    # Cosine similarity
    norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
    cosine = float(np.dot(e1, e2) / (norm1 * norm2)) if norm1 and norm2 else 0.0

    # Euclidean distance
    euclidean = float(np.linalg.norm(e1 - e2))

    # Dot product
    dot = float(np.dot(e1, e2))

    return {
        "cosine_similarity": cosine,
        "euclidean_distance": euclidean,
        "dot_product": dot,
    }
