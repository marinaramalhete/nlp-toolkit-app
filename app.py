"""
NLP Toolkit — A senior-level NLP application showcasing modern
transformer-based models for text processing tasks.

Entry point with st.navigation for multi-page routing.
"""

import streamlit as st


def home_page():
    """Render the home / landing page."""

    st.markdown(
        """
        <h1 style="text-align: center;">🤗 NLP Toolkit</h1>
        <p style="text-align: center; font-size: 1.2em; color: #666;">
            A collection of modern NLP tools powered by Transformer models
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Feature cards ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 📝 Text Summarization
            Paste any text and get an **abstractive summary** powered by
            state-of-the-art transformer models.

            | Language | Model |
            |---|---|
            | English | `BART-large-CNN` |
            | PT-BR | `PTT5 XLSum` |

            The model generates an optimal summary length automatically.
            """
        )

        st.markdown(
            """
            ### ✂️ Text Chunking
            Split long texts into meaningful segments using two strategies:

            - **Fixed-size chunking** — `RecursiveCharacterTextSplitter` with
              configurable size and overlap
            - **Semantic chunking** — Groups semantically similar sentences
              using `sentence-transformers` embeddings

            Compare both strategies side by side.
            """
        )

    with col2:
        st.markdown(
            """
            ### 🏷️ Named Entity Recognition (NER)
            Extract entities (people, organizations, locations, dates, etc.)
            from your text using **spaCy** models.

            - Rich **displaCy** visualization with colored entity spans
            - Entity table with **start/end character offsets**
            - Export results as CSV

            Supports 4 entity types in PT-BR and 18 in English.
            """
        )

        st.markdown(
            """
            ### 🔗 Semantic Similarity
            Compare two texts and measure how semantically similar they are
            using sentence-transformers.

            - **Cosine similarity** (primary metric)
            - **Euclidean distance** (secondary)
            - Language-aware models:
              `BERTimbau` (PT-BR) / `MiniLM` & `MPNet` (multilingual)
            """
        )

    st.divider()

    # ── Technical stack ──────────────────────────────────────────────────
    st.markdown("### 🛠️ Technical Stack")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    with tech_col1:
        st.markdown(
            """
            **Models & APIs**
            - Hugging Face Inference API
            - facebook/bart-large-cnn
            - recogna-nlp/ptt5-base-summ-xlsum
            - BERTimbau (PT-BR similarity)
            - sentence-transformers (MiniLM, MPNet)
            """
        )

    with tech_col2:
        st.markdown(
            """
            **NLP Libraries**
            - spaCy 3.7 (NER + displaCy)
            - LangChain (text splitters)
            - NumPy (similarity metrics)
            """
        )

    with tech_col3:
        st.markdown(
            """
            **Frontend & Deploy**
            - Streamlit 1.36+
            - Streamlit Community Cloud
            - Multi-page navigation
            """
        )

    st.divider()

    st.info(
        "⚠️ **Note:** This toolkit uses **general-purpose models** — they are not "
        "fine-tuned for any specific domain. Results may vary depending on the "
        "input text domain and language.",
        icon="ℹ️",
    )

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### About")
        st.markdown(
            """
            🤗 **NLP Toolkit** — Modern NLP tools powered by
            Transformer models.

            Find me on [GitHub](https://github.com/marinaramalhete) ✨
            """
        )
        st.markdown("---")
        st.markdown(
            """
            **How to use:**
            1. Select a tool from the navigation menu
            2. Choose your language (EN or PT-BR)
            3. Paste your text or use an example
            4. Adjust parameters and click the action button
            """
        )


# ─── Navigation ──────────────────────────────────────────────────────────────

home = st.Page(home_page, title="Home", icon="🏠", default=True)
summarization = st.Page("pages/summarization.py", title="Summarization", icon="📝")
ner = st.Page("pages/ner.py", title="Named Entity Recognition", icon="🏷️")
chunking = st.Page("pages/chunking.py", title="Text Chunking", icon="✂️")
similarity = st.Page("pages/similarity.py", title="Semantic Similarity", icon="🔗")

st.set_page_config(
    page_title="NLP Toolkit",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded",
)

nav = st.navigation([home, summarization, ner, chunking, similarity])
nav.run()
