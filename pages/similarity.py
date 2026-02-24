"""
🔗 Semantic Similarity Page
Compare two texts using multilingual sentence-transformer models.
"""

import numpy as np
import streamlit as st

from utils.config import (
    EXAMPLE_TEXTS,
    SIMILARITY_MODELS,
    get_similarity_level,
)
from utils.hf_client import get_embeddings, compute_similarity

# ─── Header ──────────────────────────────────────────────────────────────────

st.header("🔗 Semantic Similarity")
st.markdown(
    "Paste two texts and measure how **semantically similar** they are using "
    "multilingual sentence-transformer models. Compare results across models."
)

# ─── Sidebar Controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    language = st.selectbox(
        "Language",
        options=["English", "Português (PT-BR)"],
        index=0,
        help="Choose language to select the best models and example texts.",
    )

    st.markdown("---")
    st.markdown("### 🧠 Models")

    # Language-aware model list
    lang_models = SIMILARITY_MODELS[language]

    selected_models = st.multiselect(
        "Select model(s)",
        options=list(lang_models.keys()),
        default=[list(lang_models.keys())[0]],
        help="Select one or more models to compare results.",
    )

    if not selected_models:
        st.warning("Please select at least one model.")

    for name in selected_models:
        st.caption(f"**{name}:** `{lang_models[name]}`")

    st.markdown("---")

    def _load_example():
        st.session_state["sim_text_1"] = EXAMPLE_TEXTS["similarity"][language][0]
        st.session_state["sim_text_2"] = EXAMPLE_TEXTS["similarity"][language][1]

    st.button("📋 Load example texts", on_click=_load_example)


# ─── Visualization Helpers ───────────────────────────────────────────────────

def render_similarity_gauge(score: float, label: str):
    """Render a colored progress-bar-style gauge for cosine similarity."""
    level = get_similarity_level(score)
    color = level["color"]
    pct = max(0, min(100, int(score * 100)))

    st.markdown(
        f"""
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                <strong>{label}</strong>
                <span style="color: {color}; font-weight: bold; font-size: 1.2em;">{score:.4f}</span>
            </div>
            <div style="background-color: #e0e0e0; border-radius: 10px; height: 24px; overflow: hidden;">
                <div style="background-color: {color}; width: {pct}%; height: 100%;
                            border-radius: 10px; transition: width 0.5s;"></div>
            </div>
            <div style="text-align: right; color: {color}; font-size: 0.9em; margin-top: 2px;">
                {level['label_en']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─── Main Content ────────────────────────────────────────────────────────────

# Text inputs side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Text 1**")
    text_1 = st.text_area(
        "Text 1",
        height=150,
        placeholder="Paste first text here...",
        label_visibility="collapsed",
        key="sim_text_1",
    )

with col2:
    st.markdown("**Text 2**")
    text_2 = st.text_area(
        "Text 2",
        height=150,
        placeholder="Paste second text here...",
        label_visibility="collapsed",
        key="sim_text_2",
    )

# ─── Compute Similarity ─────────────────────────────────────────────────────

if st.button("🔍 Compute Similarity", type="primary", use_container_width=True):
    if not text_1.strip() or not text_2.strip():
        st.warning("Please enter text in both fields.")
    elif not selected_models:
        st.warning("Please select at least one model in the sidebar.")
    else:
        st.markdown("---")

        # Render columns based on number of models
        if len(selected_models) == 1:
            cols = [st.columns(1)[0]]
        else:
            cols = st.columns(len(selected_models))

        for col, model_label in zip(cols, selected_models):
            model_id = lang_models[model_label]

            with col:
                st.markdown(f"### {model_label}")
                st.caption(f"`{model_id}`")

                with st.spinner(f"Computing embeddings with {model_label}..."):
                    embeddings = get_embeddings(
                        [text_1, text_2],
                        model=model_id,
                    )

                if embeddings is not None and len(embeddings) == 2:
                    metrics = compute_similarity(embeddings[0], embeddings[1])

                    # Cosine similarity gauge
                    render_similarity_gauge(
                        metrics["cosine_similarity"],
                        "Cosine Similarity",
                    )

                    # Additional metrics
                    st.markdown("**Additional Metrics**")
                    mc1, mc2 = st.columns(2)
                    mc1.metric(
                        "Euclidean Distance",
                        f"{metrics['euclidean_distance']:.4f}",
                        help="Lower values = more similar",
                    )
                    mc2.metric(
                        "Dot Product",
                        f"{metrics['dot_product']:.4f}",
                        help="Higher values = more similar (for normalized vectors)",
                    )

                    # Embedding dimensions
                    st.caption(
                        f"Embedding dimensions: {embeddings[0].shape[-1]}"
                    )
                else:
                    st.error("Failed to compute embeddings.")

        # ── Interpretation ───────────────────────────────────────────
        st.markdown("---")
        with st.expander("ℹ️ How to interpret the results"):
            st.markdown(
                """
                | Score Range | Level | Interpretation |
                |---|---|---|
                | **0.8 – 1.0** | 🟢 High | Texts are semantically very similar or paraphrases |
                | **0.5 – 0.8** | 🟡 Moderate | Texts share the same topic but differ in detail |
                | **0.0 – 0.5** | 🔴 Low | Texts are about different topics |

                **Cosine Similarity** measures the angle between two embedding vectors,
                normalized to [-1, 1]. It's the standard metric for semantic textual similarity.

                **Euclidean Distance** measures the straight-line distance between vectors.
                Lower values mean the texts are more similar.

                **MiniLM** is faster (~118M params, 384 dims) while **MPNet** produces
                higher quality embeddings (~278M params, 768 dims). Comparing both helps
                validate whether the similarity signal is robust.
                """
            )

        with st.expander("ℹ️ About the models"):
            st.markdown(
                """
                **BERTimbau Portuguese STS** (`rufimelo/bert-large-portuguese-cased-sts`) —
                A BERTimbau-large model fine-tuned for Semantic Textual Similarity in Portuguese.
                Produces 1024-dimensional embeddings. Best choice for PT-BR texts.

                **paraphrase-multilingual-MiniLM-L12-v2** — A lightweight multilingual
                sentence-transformer supporting 50+ languages. Produces 384-dimensional
                embeddings. Best speed/quality tradeoff.

                **paraphrase-multilingual-mpnet-base-v2** — A higher-quality multilingual
                model producing 768-dimensional embeddings. Better accuracy but slower.

                The multilingual models are from [sentence-transformers](https://www.sbert.net/)
                and work well across languages. For Portuguese specifically, the BERTimbau
                model provides stronger differentiation between similar and dissimilar texts.
                """
            )
