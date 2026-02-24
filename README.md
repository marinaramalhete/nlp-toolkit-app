# 🤗 NLP Toolkit

A multi-feature NLP application built with **Streamlit** and powered by modern **Transformer** models. Supports both **English** and **Portuguese (PT-BR)**.

---

## Features

| Feature | Description | Models / Libraries |
|---|---|---|
| **📝 Text Summarization** | Abstractive summarization via Hugging Face Inference API | `BART-large-CNN` (EN), `PTT5-base-summ-xlsum` (PT-BR) |
| **🏷️ Named Entity Recognition** | Entity extraction with displaCy visualization and CSV export | `spaCy en_core_web_sm` (18 entity types), `pt_core_news_sm` (4 types) |
| **✂️ Text Chunking** | Fixed-size and semantic chunking with side-by-side comparison | `LangChain RecursiveCharacterTextSplitter`, HF embeddings |
| **🔗 Semantic Similarity** | Cosine similarity between texts with language-aware model selection | `BERTimbau` (PT-BR), `MiniLM` / `MPNet` (multilingual) |

## Architecture

```
app.py                  # Entry point — st.navigation multi-page routing
pages/
  summarization.py      # HF Inference API summarization
  ner.py                # spaCy NER + displaCy rendering
  chunking.py           # Fixed-size & semantic chunking
  similarity.py         # Sentence-transformer similarity
utils/
  config.py             # Centralized models, colors, thresholds, example texts
  hf_client.py          # Shared InferenceClient with retry logic
  spacy_models.py       # Cached spaCy model loading
```

## Tech Stack

- **Streamlit** — Multi-page UI with `st.navigation`
- **Hugging Face Inference API** — Serverless model inference (free tier)
- **spaCy 3.7** — NER pipelines + displaCy visualization
- **LangChain** — Text splitters for chunking strategies
- **NumPy / Pandas** — Similarity metrics and data export

## Getting Started

### Prerequisites

- Python 3.11+
- A [Hugging Face](https://huggingface.co/settings/tokens) API token (free)

### Installation

```bash
git clone https://github.com/marinaramalhete/NLP-Text-Summary-App.git
cd NLP-Text-Summary-App

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.streamlit/secrets.toml` file with your HF token:

```toml
HF_TOKEN = "hf_..."
```

### Running

```bash
streamlit run app.py
```

## Deployment

This app is designed for **Streamlit Community Cloud**. To deploy your own instance:

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Add `HF_TOKEN` in the app's **Secrets** settings
4. Deploy

## Author

**Marina Ramalhete Masid** — [GitHub](https://github.com/marinaramalhete) · [LinkedIn](https://www.linkedin.com/in/marinaramalhete/)
