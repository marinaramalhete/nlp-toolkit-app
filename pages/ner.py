"""
🏷️ Named Entity Recognition (NER) Page
Entity extraction using spaCy with displaCy visualization.
"""

import pandas as pd
import streamlit as st
from spacy import displacy

from utils.config import (
    EXAMPLE_TEXTS,
    NER_COLORS,
    NER_ENTITY_DESCRIPTIONS,
    SPACY_MODELS,
)
from utils.spacy_models import load_spacy_model

# ─── Header ──────────────────────────────────────────────────────────────────

st.header("🏷️ Named Entity Recognition")
st.markdown(
    "Paste your text below and extract **named entities** (people, organizations, "
    "locations, dates, etc.) with their character spans. Powered by spaCy."
)

# ─── Sidebar Controls ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    language = st.selectbox(
        "Language",
        options=list(SPACY_MODELS.keys()),
        index=0,
        help="Select the language of your input text.",
    )

    model_name = SPACY_MODELS[language]
    st.caption(f"**Model:** `{model_name}`")

    st.markdown("---")

    # Entity type legend
    st.markdown("### 🎨 Entity Types")

    if language == "Português (PT-BR)":
        entity_types = ["PER", "LOC", "ORG", "MISC"]
    else:
        entity_types = [
            "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC",
            "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
            "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY",
            "ORDINAL", "CARDINAL",
        ]

    for ent_type in entity_types:
        color = NER_COLORS.get(ent_type, "#ddd")
        desc = NER_ENTITY_DESCRIPTIONS.get(ent_type, "")
        st.markdown(
            f'<span style="background-color: {color}; padding: 2px 8px; '
            f'border-radius: 4px; font-size: 0.85em; font-weight: bold;">'
            f"{ent_type}</span> {desc}",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    def _load_example():
        st.session_state["ner_input"] = EXAMPLE_TEXTS["ner"][language]

    st.button("📋 Load example text", on_click=_load_example)

# ─── Main Content ────────────────────────────────────────────────────────────

input_text = st.text_area(
    "Input text",
    height=200,
    placeholder="Paste your text here...",
    label_visibility="collapsed",
    key="ner_input",
)

# ─── Extract Entities ────────────────────────────────────────────────────────

if st.button("🔍 Extract Entities", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Processing text with spaCy..."):
            nlp = load_spacy_model(language)
            doc = nlp(input_text)

        if not doc.ents:
            st.info("No named entities found in this text.")
        else:
            # ── displaCy Visualization ───────────────────────────────
            st.markdown("### 🖼️ Entity Visualization")

            html = displacy.render(
                doc,
                style="ent",
                page=False,
                options={"colors": NER_COLORS},
            )
            # Wrap in a styled div for better rendering
            st.markdown(
                f'<div style="line-height: 2.5; font-size: 1.1em;">{html}</div>',
                unsafe_allow_html=True,
            )

            # ── Entity Table ─────────────────────────────────────────
            st.markdown("### 📋 Entity Details")

            entities_data = []
            for ent in doc.ents:
                entities_data.append(
                    {
                        "Entity": ent.text,
                        "Type": ent.label_,
                        "Description": NER_ENTITY_DESCRIPTIONS.get(
                            ent.label_, "Unknown"
                        ),
                        "Start": ent.start_char,
                        "End": ent.end_char,
                    }
                )

            df = pd.DataFrame(entities_data)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Entity": st.column_config.TextColumn("Entity", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Description": st.column_config.TextColumn(
                        "Description", width="large"
                    ),
                    "Start": st.column_config.NumberColumn("Start Char", width="small"),
                    "End": st.column_config.NumberColumn("End Char", width="small"),
                },
            )

            # ── Summary Metrics ──────────────────────────────────────
            st.markdown("### 📊 Summary")

            unique_types = df["Type"].nunique()
            total_entities = len(df)

            m1, m2 = st.columns(2)
            m1.metric("Total Entities", total_entities)
            m2.metric("Unique Entity Types", unique_types)

            # Type distribution
            st.bar_chart(
                df["Type"].value_counts(),
                color="#aa9cfc",
            )

            # ── Export ───────────────────────────────────────────────
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download entities as CSV",
                data=csv,
                file_name="entities.csv",
                mime="text/csv",
            )

            # ── Model Info ───────────────────────────────────────────
            with st.expander("ℹ️ About this model"):
                if language == "English":
                    st.markdown(
                        """
                        **en_core_web_sm** is a small English spaCy pipeline trained on
                        OntoNotes 5.0, supporting **18 entity types**.

                        - Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer
                        - NER F-score: ~0.85
                        - Size: ~12 MB
                        - [spaCy model page](https://spacy.io/models/en#en_core_web_sm)
                        """
                    )
                else:
                    st.markdown(
                        """
                        **pt_core_news_sm** is a small Portuguese spaCy pipeline trained on
                        WikiNER and UD Portuguese Bosque, supporting **4 entity types**
                        (PER, LOC, ORG, MISC).

                        - Components: tok2vec, morphologizer, parser, senter, ner, attribute_ruler, lemmatizer
                        - NER F-score: ~0.78
                        - Size: ~15 MB
                        - [spaCy model page](https://spacy.io/models/pt#pt_core_news_sm)
                        """
                    )
