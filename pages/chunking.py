"""
✂️ Text Chunking Page
Fixed-size and semantic chunking with side-by-side comparison.
"""

import streamlit as st
import numpy as np

from utils.config import EXAMPLE_TEXTS, CHUNK_COLORS, EMBEDDING_MODEL
from utils.hf_client import get_embeddings

# ─── Header ──────────────────────────────────────────────────────────────────

st.header("✂️ Text Chunking")
st.markdown(
    "Split your text into meaningful segments using **fixed-size** or "
    "**semantic** chunking strategies. Compare both approaches side by side."
)

# ─── Sidebar Controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    language = st.selectbox(
        "Language",
        options=["English", "Português (PT-BR)"],
        index=0,
        help="Select the language of your input text.",
    )

    strategy = st.selectbox(
        "Chunking Strategy",
        options=["Fixed-size with Overlap", "Semantic", "Both (compare)"],
        index=2,
        help="Choose how to split the text.",
    )

    st.markdown("---")

    # Fixed-size settings
    if strategy in ["Fixed-size with Overlap", "Both (compare)"]:
        st.markdown("### 📐 Fixed-size Settings")
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=100,
            max_value=2000,
            value=500,
            step=50,
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
        )
        overlap_valid = chunk_overlap < chunk_size
        if not overlap_valid:
            st.warning("⚠️ Overlap must be smaller than chunk size.")

    # Semantic settings
    if strategy in ["Semantic", "Both (compare)"]:
        st.markdown("### 🧠 Semantic Settings")
        st.caption(f"**Embedding model:** `{EMBEDDING_MODEL}`")
        breakpoint_threshold = st.slider(
            "Breakpoint sensitivity",
            min_value=10,
            max_value=95,
            value=70,
            step=5,
            help=(
                "Higher values = fewer, larger chunks. "
                "Controls how aggressively the text is split: higher values "
                "require a bigger similarity drop to create a new chunk."
            ),
        )

    st.markdown("---")

    def _load_example():
        st.session_state["chunk_input"] = EXAMPLE_TEXTS["chunking"][language]

    st.button("📋 Load example text", on_click=_load_example)


# ─── Chunking Functions ─────────────────────────────────────────────────────

def fixed_size_chunk(text: str, size: int, overlap: int) -> list[str]:
    """Split text using RecursiveCharacterTextSplitter."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([text])
    return [doc.page_content for doc in docs]


def semantic_chunk(text: str, threshold_percentile: int) -> list[str]:
    """
    Semantic chunking using sentence embeddings from HF Inference API.
    Groups consecutive sentences when their cosine similarity is above
    the threshold (determined by percentile of all pairwise similarities).
    """
    import re

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return [text]

    # Get embeddings
    embeddings = get_embeddings(sentences, model=EMBEDDING_MODEL)
    if embeddings is None:
        st.error("Failed to get embeddings for semantic chunking.")
        return [text]

    # Compute cosine similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        e1, e2 = embeddings[i], embeddings[i + 1]
        norm1, norm2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if norm1 > 0 and norm2 > 0:
            sim = float(np.dot(e1, e2) / (norm1 * norm2))
        else:
            sim = 0.0
        similarities.append(sim)

    # Determine breakpoint threshold
    threshold = float(np.percentile(similarities, 100 - threshold_percentile))

    # Build chunks: break where similarity drops below threshold
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ─── Visualization Helper ───────────────────────────────────────────────────

def render_chunks(chunks: list[str], label: str):
    """Render chunks with colored backgrounds and metrics."""
    st.markdown(f"#### {label}")

    # Metrics row
    sizes = [len(c) for c in chunks]
    m1, m2, m3 = st.columns(3)
    m1.metric("Chunks", len(chunks))
    m2.metric("Avg size (chars)", f"{int(np.mean(sizes)):,}" if sizes else "0")
    m3.metric("Total chars", f"{sum(sizes):,}" if sizes else "0")

    # Render colored chunks
    for i, chunk in enumerate(chunks):
        color = CHUNK_COLORS[i % len(CHUNK_COLORS)]
        st.markdown(
            f'<div style="background-color: {color}; padding: 12px 16px; '
            f'border-radius: 8px; margin-bottom: 8px; border-left: 4px solid '
            f'{color.replace("FF", "AA").replace("C1", "80")};">'
            f'<strong>Chunk {i + 1}</strong> '
            f'<span style="color: #888; font-size: 0.85em;">({len(chunk)} chars)</span>'
            f'<br/>{chunk}</div>',
            unsafe_allow_html=True,
        )

    # Size distribution chart
    if len(chunks) > 1:
        import pandas as pd
        chart_df = pd.DataFrame({
            "Chunk": [f"Chunk {i+1}" for i in range(len(chunks))],
            "Size (chars)": sizes,
        }).set_index("Chunk")
        st.bar_chart(chart_df, color="#7aecec")


# ─── Main Content ────────────────────────────────────────────────────────────

input_text = st.text_area(
    "Input text",
    height=250,
    placeholder="Paste your text here...",
    label_visibility="collapsed",
    key="chunk_input",
)

# ─── Process Chunks ─────────────────────────────────────────────────────────

if st.button("✂️ Chunk Text", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to chunk.")
    elif len(input_text) < 50:
        st.warning("Text is too short for meaningful chunking. Please provide more content.")
    else:
        # Guard: don't run fixed-size chunking with invalid overlap
        needs_fixed = strategy in ["Fixed-size with Overlap", "Both (compare)"]
        if needs_fixed and not overlap_valid:
            st.error("Cannot chunk: overlap must be smaller than chunk size. Please adjust the settings.")
        elif strategy == "Both (compare)":
            col_fixed, col_semantic = st.columns(2)

            with col_fixed:
                with st.spinner("Running fixed-size chunking..."):
                    fixed_chunks = fixed_size_chunk(input_text, chunk_size, chunk_overlap)
                render_chunks(fixed_chunks, "📐 Fixed-size Chunks")

            with col_semantic:
                with st.spinner("Running semantic chunking (calling HF API)..."):
                    sem_chunks = semantic_chunk(input_text, breakpoint_threshold)
                render_chunks(sem_chunks, "🧠 Semantic Chunks")

        elif strategy == "Fixed-size with Overlap":
            with st.spinner("Running fixed-size chunking..."):
                chunks = fixed_size_chunk(input_text, chunk_size, chunk_overlap)
            render_chunks(chunks, "📐 Fixed-size Chunks")

        else:  # Semantic
            with st.spinner("Running semantic chunking (calling HF API)..."):
                chunks = semantic_chunk(input_text, breakpoint_threshold)
            render_chunks(chunks, "🧠 Semantic Chunks")

        # ── Model Info ───────────────────────────────────────────────
        with st.expander("ℹ️ About chunking strategies"):
            st.markdown(
                f"""
                **Fixed-size chunking** uses `RecursiveCharacterTextSplitter` from LangChain,
                which tries to split on natural boundaries (`\\n\\n`, `\\n`, `.`, ` `) while
                respecting the configured chunk size and overlap.

                **Semantic chunking** computes sentence embeddings using
                `{EMBEDDING_MODEL}` via the Hugging Face Inference API, then measures
                cosine similarity between consecutive sentences. A chunk boundary is
                inserted where the similarity drops below the {breakpoint_threshold}th
                percentile threshold.

                Semantic chunking produces topic-coherent segments, which is ideal for
                **RAG pipelines** where downstream retrieval benefits from semantically
                focused passages.
                """
            )
