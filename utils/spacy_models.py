"""
Cached spaCy model loading for the NLP Toolkit.
"""

import spacy
import streamlit as st

from utils.config import SPACY_MODELS


@st.cache_resource
def load_spacy_model(language: str) -> spacy.language.Language:
    """
    Load and cache a spaCy model for the given language key.
    Uses @st.cache_resource so the model is loaded only once per session.
    """
    model_name = SPACY_MODELS[language]
    return spacy.load(model_name)
