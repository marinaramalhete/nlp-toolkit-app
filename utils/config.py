"""
Centralized configuration for the NLP Toolkit.
All model names, colors, example texts, and constants live here.
"""

# ─── Hugging Face Models ─────────────────────────────────────────────────────

SUMMARIZATION_MODELS = {
    "English": "facebook/bart-large-cnn",
    "Português (PT-BR)": "recogna-nlp/ptt5-base-summ-xlsum",
}

SIMILARITY_MODELS = {
    "English": {
        "MiniLM (fast, multilingual)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "MPNet (best quality, multilingual)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    },
    "Português (PT-BR)": {
        "BERTimbau (Portuguese STS)": "rufimelo/bert-large-portuguese-cased-sts",
        "MiniLM (multilingual)": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "MPNet (multilingual)": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    },
}

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ─── spaCy Models ────────────────────────────────────────────────────────────

SPACY_MODELS = {
    "English": "en_core_web_sm",
    "Português (PT-BR)": "pt_core_news_sm",
}

# ─── NER Entity Colors ───────────────────────────────────────────────────────

NER_COLORS = {
    # Portuguese (HAREM/WikiNER)
    "PER": "#aa9cfc",
    "LOC": "#feca74",
    "ORG": "#7aecec",
    "MISC": "#c887fb",
    # English (OntoNotes)
    "PERSON": "#aa9cfc",
    "NORP": "#c887fb",
    "FAC": "#ddd",
    "GPE": "#feca74",
    "EVENT": "#ffb347",
    "WORK_OF_ART": "#f0d0ff",
    "LAW": "#d4edda",
    "LANGUAGE": "#e8daef",
    "DATE": "#bfe1d9",
    "TIME": "#bfe1d9",
    "PERCENT": "#e4e7eb",
    "MONEY": "#e4e7eb",
    "QUANTITY": "#e4e7eb",
    "ORDINAL": "#e8daef",
    "CARDINAL": "#e8daef",
    "PRODUCT": "#ddd",
}

NER_ENTITY_DESCRIPTIONS = {
    # Portuguese
    "PER": "Person — names of people",
    "LOC": "Location — cities, countries, geographical features",
    "ORG": "Organization — companies, agencies, institutions",
    "MISC": "Miscellaneous — events, nationalities, other",
    # English
    "PERSON": "Person — people, including fictional",
    "NORP": "Nationalities, religious or political groups",
    "FAC": "Facility — buildings, airports, highways",
    "GPE": "Geo-Political Entity — countries, cities, states",
    "EVENT": "Named events — hurricanes, wars, sports events",
    "WORK_OF_ART": "Titles of books, songs, etc.",
    "LAW": "Named documents made into laws",
    "LANGUAGE": "Any named language",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "PERCENT": "Percentage (including %)",
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements (weight, distance, etc.)",
    "ORDINAL": '"first", "second", etc.',
    "CARDINAL": "Numerals that do not fall under another type",
    "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
}

# ─── Chunk Visualization Colors ──────────────────────────────────────────────

CHUNK_COLORS = [
    "#FFDDC1", "#C1E1FF", "#D4F0C1", "#FFC1E3",
    "#E1C1FF", "#C1FFE1", "#FFE8C1", "#C1FFFA",
    "#FFC1C1", "#C1C1FF", "#E8FFC1", "#FFC1FA",
]

# ─── Example Texts ───────────────────────────────────────────────────────────

EXAMPLE_TEXTS = {
    "summarization": {
        "English": (
            "Artificial intelligence (AI) has transformed numerous industries over the past decade. "
            "From healthcare to finance, AI systems are now capable of performing tasks that were once "
            "thought to require human intelligence. Machine learning, a subset of AI, enables computers "
            "to learn from data without being explicitly programmed. Deep learning, which uses neural "
            "networks with many layers, has achieved remarkable results in image recognition, natural "
            "language processing, and game playing. However, the rapid advancement of AI also raises "
            "important ethical questions about privacy, bias, job displacement, and the concentration "
            "of power in the hands of a few technology companies. Researchers and policymakers are "
            "working together to establish guidelines and regulations that ensure AI development "
            "benefits society as a whole while minimizing potential risks."
        ),
        "Português (PT-BR)": (
            "A inteligência artificial (IA) transformou diversas indústrias na última década. "
            "Da saúde às finanças, os sistemas de IA agora são capazes de realizar tarefas que antes "
            "se pensava exigirem inteligência humana. O aprendizado de máquina, um subcampo da IA, "
            "permite que computadores aprendam a partir de dados sem serem explicitamente programados. "
            "O aprendizado profundo, que utiliza redes neurais com muitas camadas, alcançou resultados "
            "notáveis em reconhecimento de imagem, processamento de linguagem natural e jogos. "
            "No entanto, o rápido avanço da IA também levanta questões éticas importantes sobre "
            "privacidade, viés, deslocamento de empregos e a concentração de poder nas mãos de "
            "poucas empresas de tecnologia. Pesquisadores e formuladores de políticas estão "
            "trabalhando juntos para estabelecer diretrizes e regulamentações que garantam que o "
            "desenvolvimento da IA beneficie a sociedade como um todo, minimizando riscos potenciais."
        ),
    },
    "ner": {
        "English": (
            "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, "
            "in Los Altos, California. The company is headquartered in Cupertino and reported revenue "
            "of $394.3 billion in 2022. Tim Cook has been the CEO since August 2011. Apple's products "
            "include the iPhone, iPad, and MacBook, which are sold in over 175 countries worldwide."
        ),
        "Português (PT-BR)": (
            "A Petrobras, fundada em 3 de outubro de 1953 pelo presidente Getúlio Vargas, é uma das "
            "maiores empresas de energia do mundo. Com sede no Rio de Janeiro, a empresa opera em "
            "diversos países da América Latina, África e Europa. Em 2023, a Petrobras reportou um "
            "lucro líquido de R$ 124,6 bilhões. O atual presidente da companhia, Jean Paul Prates, "
            "assumiu o cargo em janeiro de 2023."
        ),
    },
    "chunking": {
        "English": (
            "The history of computing is a fascinating journey that spans several centuries. "
            "The earliest known computing device is the abacus, which was used in ancient civilizations "
            "for basic arithmetic operations.\n\n"
            "In the 17th century, Blaise Pascal invented the Pascaline, one of the first mechanical "
            "calculators. Later, Charles Babbage designed the Analytical Engine, which is considered "
            "the first general-purpose computer concept.\n\n"
            "The modern era of computing began in the 1940s with the development of electronic "
            "computers like ENIAC and UNIVAC. These machines were enormous, filling entire rooms, "
            "and were primarily used for military and scientific calculations.\n\n"
            "The invention of the transistor in 1947 revolutionized computing by making it possible "
            "to build smaller, faster, and more reliable computers. This led to the development of "
            "mainframe computers in the 1950s and 1960s.\n\n"
            "The personal computer revolution began in the 1970s with machines like the Apple II "
            "and the IBM PC. The introduction of graphical user interfaces in the 1980s made "
            "computers accessible to the general public.\n\n"
            "The rise of the internet in the 1990s transformed computing from a standalone activity "
            "into a connected experience. Today, cloud computing, artificial intelligence, and "
            "quantum computing represent the cutting edge of the field."
        ),
        "Português (PT-BR)": (
            "A história da computação é uma jornada fascinante que abrange vários séculos. "
            "O dispositivo de computação mais antigo conhecido é o ábaco, utilizado em civilizações "
            "antigas para operações aritméticas básicas.\n\n"
            "No século XVII, Blaise Pascal inventou a Pascalina, uma das primeiras calculadoras "
            "mecânicas. Posteriormente, Charles Babbage projetou a Máquina Analítica, considerada "
            "o primeiro conceito de computador de uso geral.\n\n"
            "A era moderna da computação começou na década de 1940 com o desenvolvimento de "
            "computadores eletrônicos como o ENIAC e o UNIVAC. Essas máquinas eram enormes, "
            "ocupando salas inteiras, e eram usadas principalmente para cálculos militares.\n\n"
            "A invenção do transistor em 1947 revolucionou a computação, tornando possível "
            "construir computadores menores, mais rápidos e confiáveis. Isso levou ao desenvolvimento "
            "dos mainframes nas décadas de 1950 e 1960.\n\n"
            "A revolução do computador pessoal começou na década de 1970 com máquinas como o Apple II "
            "e o IBM PC. A introdução de interfaces gráficas nos anos 1980 tornou os computadores "
            "acessíveis ao público em geral.\n\n"
            "A ascensão da internet nos anos 1990 transformou a computação de uma atividade isolada "
            "em uma experiência conectada. Hoje, computação em nuvem, inteligência artificial e "
            "computação quântica representam a vanguarda da área."
        ),
    },
    "similarity": {
        "English": (
            "The cat sat on the mat.",
            "A feline was resting on the rug.",
        ),
        "Português (PT-BR)": (
            "O gato sentou no tapete.",
            "Um felino estava descansando sobre o tapete.",
        ),
    },
}

# ─── Similarity Interpretation Thresholds ────────────────────────────────────

SIMILARITY_LEVELS = {
    "high": {"min": 0.8, "label_en": "High Similarity", "label_pt": "Alta Similaridade", "color": "#28a745"},
    "moderate": {"min": 0.5, "label_en": "Moderate Similarity", "label_pt": "Similaridade Moderada", "color": "#ffc107"},
    "low": {"min": 0.0, "label_en": "Low Similarity", "label_pt": "Baixa Similaridade", "color": "#dc3545"},
}


def get_similarity_level(score: float) -> dict:
    """Return the similarity level dict for a given cosine similarity score."""
    if score >= SIMILARITY_LEVELS["high"]["min"]:
        return SIMILARITY_LEVELS["high"]
    elif score >= SIMILARITY_LEVELS["moderate"]["min"]:
        return SIMILARITY_LEVELS["moderate"]
    else:
        return SIMILARITY_LEVELS["low"]
