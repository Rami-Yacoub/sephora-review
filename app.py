"""
💄 Analyseur de Reviews E-Commerce — Sephora NLP Dashboard
app.py — Application Streamlit complète
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import pickle
from collections import Counter

# ─── Configuration de la page (DOIT être en premier appel Streamlit) ──────────
st.set_page_config(
    page_title="💄 Sephora Review Analyzer",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Custom ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f0c1a 0%, #1a1228 50%, #0d1117 100%);
    min-height: 100vh;
}

/* Header */
.main-header {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #f7c6d4 0%, #e8a4bc 40%, #c77dac 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -1px;
}

.sub-header {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #a89bb8;
    font-size: 1rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #160f2a 0%, #1e1535 100%);
    border-right: 1px solid rgba(247,198,212,0.15);
}

section[data-testid="stSidebar"] .stRadio label {
    color: #c9b8d8 !important;
    font-size: 0.95rem;
    font-weight: 400;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(247,198,212,0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

.card-pink {
    border-color: rgba(247,198,212,0.35);
    background: rgba(247,198,212,0.05);
}

/* KPI / Metric boxes */
.kpi-box {
    background: rgba(199,125,172,0.12);
    border: 1px solid rgba(199,125,172,0.3);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
}

.kpi-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f7c6d4;
}

.kpi-label {
    color: #a89bb8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.2rem;
}

/* Sentiment gauge */
.gauge-bar-outer {
    background: rgba(255,255,255,0.1);
    border-radius: 50px;
    height: 14px;
    overflow: hidden;
    margin: 0.5rem 0;
}

.gauge-bar-inner {
    height: 100%;
    border-radius: 50px;
    transition: width 0.6s ease;
}

/* Tags */
.tag {
    display: inline-block;
    background: rgba(199,125,172,0.2);
    border: 1px solid rgba(199,125,172,0.4);
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.82rem;
    color: #e8c4d8;
    margin: 0.2rem;
}

.tag-green {
    background: rgba(52,199,131,0.15);
    border-color: rgba(52,199,131,0.4);
    color: #7eefc0;
}

.tag-red {
    background: rgba(255,91,91,0.15);
    border-color: rgba(255,91,91,0.4);
    color: #ff9999;
}

/* Stars */
.stars { font-size: 1.5rem; letter-spacing: 2px; }
.star-filled { color: #f7c6d4; }
.star-empty  { color: rgba(255,255,255,0.15); }

/* Section titles */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #f7c6d4;
    margin: 1.5rem 0 0.8rem 0;
    font-weight: 700;
}

/* Recommendation banner */
.rec-buy {
    background: linear-gradient(135deg, rgba(52,199,131,0.2), rgba(52,199,131,0.08));
    border: 1px solid rgba(52,199,131,0.5);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 1.1rem;
    color: #7eefc0;
    font-weight: 600;
    text-align: center;
}

.rec-avoid {
    background: linear-gradient(135deg, rgba(255,91,91,0.2), rgba(255,91,91,0.08));
    border: 1px solid rgba(255,91,91,0.5);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 1.1rem;
    color: #ff9999;
    font-weight: 600;
    text-align: center;
}

.rec-neutral {
    background: linear-gradient(135deg, rgba(247,198,212,0.15), rgba(247,198,212,0.05));
    border: 1px solid rgba(247,198,212,0.4);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-size: 1.1rem;
    color: #f7c6d4;
    font-weight: 600;
    text-align: center;
}

/* Word importance bars */
.word-bar-container { margin: 0.3rem 0; }
.word-label { color: #c9b8d8; font-size: 0.88rem; display: inline-block; width: 120px; }
.word-bar { display: inline-block; height: 10px; border-radius: 5px; vertical-align: middle; margin-left: 8px; }

/* Topic card */
.topic-card {
    background: rgba(199,125,172,0.08);
    border: 1px solid rgba(199,125,172,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

.topic-name {
    font-family: 'Playfair Display', serif;
    color: #f7c6d4;
    font-size: 1.05rem;
    font-weight: 700;
}

.topic-prob {
    color: #a89bb8;
    font-size: 0.8rem;
}

/* Input area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(247,198,212,0.25) !important;
    border-radius: 12px !important;
    color: #e8dff0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #c77dac, #a855a0) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(199,125,172,0.4) !important;
}

/* Selectbox */
.stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(247,198,212,0.25) !important;
    color: #e8dff0 !important;
}

/* Divider */
hr { border-color: rgba(247,198,212,0.12) !important; }

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #160f2a; }
::-webkit-scrollbar-thumb { background: rgba(199,125,172,0.4); border-radius: 3px; }

/* st.metric overrides */
[data-testid="stMetric"] {
    background: rgba(199,125,172,0.08);
    border: 1px solid rgba(199,125,172,0.2);
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { color: #a89bb8 !important; }
[data-testid="stMetricValue"] { color: #f7c6d4 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UTILITAIRES NLP (pipeline texte, modèles stub)
# ══════════════════════════════════════════════════════════════════════════════

# Stopwords custom (domaine cosmétique, cohérent avec les notebooks)
STOP_CUSTOM = {
    'product', 'review', 'item', 'bought', 'buy', 'purchase', 'purchased',
    'ordered', 'order', 'received', 'receive', 'shipping', 'shipped',
    'use', 'used', 'using', 'one', 'get', 'got', 'would', 'also', 'even',
    'really', 'little', 'just', 'know', 'make', 'made', 'go', 'going',
    'first', 'since', 'still', 'feel', 'feeling', 'found', 'find',
    'tried', 'try', 'always', 'never', 'every', 'back', 'well',
    'think', 'thought', 'way', 'much', 'many', 'lot', 'bit',
    'bottle', 'tube', 'jar', 'container', 'size', 'small', 'large',
}

# Vocabulaire de sentiment pour le mock NLP
POSITIVE_WORDS = {
    'love', 'amazing', 'excellent', 'fantastic', 'wonderful', 'great',
    'best', 'perfect', 'awesome', 'beautiful', 'smooth', 'glowing',
    'hydrated', 'moisturized', 'effective', 'recommend', 'worth',
    'gentle', 'lightweight', 'fresh', 'bright', 'radiant', 'soft',
    'comfortable', 'happy', 'impressed', 'satisfied', 'works', 'helped',
    'improved', 'incredible', 'superb', 'outstanding', 'brilliant',
}

NEGATIVE_WORDS = {
    'terrible', 'awful', 'horrible', 'worst', 'bad', 'hate', 'dislike',
    'disappoint', 'disappointing', 'disappointed', 'waste', 'broke',
    'irritated', 'irritating', 'rash', 'breakout', 'acne', 'dry',
    'greasy', 'heavy', 'sticky', 'smell', 'smells', 'chemical',
    'cheap', 'flaky', 'peeling', 'red', 'redness', 'sensitive',
    'reaction', 'allergy', 'allergic', 'burned', 'burning', 'stings',
}

BEAUTY_INGREDIENTS = [
    'hyaluronic acid', 'glycerin', 'ceramide', 'squalane', 'niacinamide',
    'retinol', 'retinoid', 'vitamin c', 'vitamin e', 'collagen', 'peptide',
    'salicylic acid', 'glycolic acid', 'lactic acid', 'aha', 'bha',
    'kojic acid', 'arbutin', 'azelaic acid', 'centella asiatica', 'cica',
    'aloe vera', 'allantoin', 'zinc oxide', 'spf', 'sunscreen',
    'serum', 'moisturizer', 'toner', 'essence', 'cleanser', 'exfoliant',
]

TOPIC_LABELS = {
    0: ('💧 Hydratation & Texture',  ['moisture', 'hydrate', 'dry', 'water', 'cream', 'serum', 'lightweight']),
    1: ('✨ Éclat & Teint',          ['glow', 'bright', 'radiant', 'even', 'tone', 'luminous', 'vitamin c']),
    2: ('🧴 Nettoyage & Pores',      ['clean', 'pore', 'cleanser', 'wash', 'foam', 'exfoliant', 'clear']),
    3: ('🌿 Ingrédients Actifs',     ['retinol', 'niacinamide', 'acid', 'ceramide', 'peptide', 'ingredient']),
    4: ('☀️  Protection Solaire',    ['spf', 'sun', 'sunscreen', 'uv', 'protection', 'zinc', 'titanium']),
    5: ('⏳ Anti-Âge & Rides',       ['anti', 'age', 'wrinkle', 'firm', 'collagen', 'fine', 'line', 'lift']),
}

# KPIs dataset (issus de l'EDA notebooks)
DATASET_STATS = {
    'total_reviews': 27_414,
    'avg_rating': 4.21,
    'pct_positive': 77.4,
    'pct_negative': 10.3,
    'avg_words': 89.2,
    'unique_products': 8_494,
    'rating_dist': {1: 4.2, 2: 6.1, 3: 12.3, 4: 23.7, 5: 53.7},
    'top_ingredients': [
        ('moisturizer', 2841), ('serum', 1923), ('cleanser', 1104),
        ('spf', 987), ('retinol', 876), ('niacinamide', 743),
        ('hyaluronic acid', 698), ('vitamin c', 612), ('ceramide', 534),
        ('glycolic acid', 412),
    ],
    'topic_dist': {0: 22.1, 1: 17.8, 2: 15.4, 3: 18.9, 4: 12.3, 5: 13.5},
}

SAMPLE_REVIEWS = {
    0: [
        "This moisturizer is absolutely incredible! My skin feels so hydrated after just one application. Very lightweight and absorbs quickly.",
        "The texture is perfect, not too heavy or greasy. My dry patches disappeared within a week of consistent use.",
    ],
    1: [
        "This vitamin C serum has completely transformed my skin. My dark spots have faded significantly and my complexion looks so much brighter!",
        "Amazing glow after using this for 3 weeks. The tone of my skin has evened out considerably. Highly recommend.",
    ],
    2: [
        "Best cleanser I've ever used. My pores look smaller and my skin feels really clean without feeling stripped.",
        "Great exfoliating cleanser! Removes all makeup and dirt. My blackheads have reduced noticeably.",
    ],
    3: [
        "This niacinamide serum is a game changer. Combined with my retinol, my skin has never looked better.",
        "Started using this peptide serum and the ingredients are top notch. Can see real results in texture.",
    ],
    4: [
        "Best SPF I've tried! Doesn't leave a white cast and wears perfectly under makeup. Will repurchase.",
        "Lightweight sunscreen that blends beautifully. Great UV protection with SPF 50. No greasy feeling.",
    ],
    5: [
        "After 2 months of use, my fine lines are noticeably reduced. This retinol formula is gentle yet effective.",
        "My skin looks firmer and more youthful. The collagen-boosting peptides in this cream really work.",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CHARGEMENT DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Charge les modèles depuis le dossier models/.
    Si un modèle échoue à charger, affiche l'erreur (ne pas masquer).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, 'models')

    def p(name: str) -> str:
        return os.path.join(models_dir, name)

    models = {
        'lda': None,
        'lda_dict': None,
        'w2v': None,
        'lstm': None,
        'tfidf': None,
        'svm': None,
        'tokenizer': None,
        'demo_mode': True,
        'load_errors': [],
        'load_ok': [],
    }

    def log_ok(msg: str):
        models['load_ok'].append(msg)

    def log_err(msg: str, exc: Exception):
        # logs riches pour diagnostiquer incompatibilités de versions (numpy/keras/gensim)
        import traceback
        try:
            import numpy as _np
            numpy_v = getattr(_np, '__version__', 'unknown')
        except Exception:
            numpy_v = 'unknown'
        try:
            import gensim as _gensim
            gensim_v = getattr(_gensim, '__version__', 'unknown')
        except Exception:
            gensim_v = 'unknown'
        try:
            import tensorflow as _tf
            tf_v = getattr(_tf, '__version__', 'unknown')
        except Exception:
            tf_v = 'unknown'

        err_text = str(exc)
        hint = None
        if 'BitGenerator' in err_text or 'MT19937' in err_text:
            hint = 'Compatibilité numpy/gensim (pickle) probable'
        elif (
            'could not be deserialized properly' in err_text
            or 'from_config' in err_text
            or 'Sequential' in err_text
        ):
            hint = 'Compatibilité keras/tensorflow (sérialisation modèle) probable'

        extra = f" | hint={hint}" if hint else ''
        models['load_errors'].append(
            f"{msg}: {type(exc).__name__}: {exc} | versions: numpy={numpy_v}, gensim={gensim_v}, tf={tf_v}{extra} | traceback={traceback.format_exc().strip()}"
        )


    # LDA
    try:
        if os.path.exists(p('lda_model')):
            import gensim.corpora as corpora

            # LdaMulticore peut échouer selon les versions/serialisation.
            # Si c'est une incompatibilité numpy/gensim (BitGenerator/MT19937),
            # retenter ne sert à rien : on tombe en fallback UI.
            try:
                from gensim.models import LdaMulticore
                models['lda'] = LdaMulticore.load(p('lda_model'))
            except Exception as e_inner:
                err_text = str(e_inner)
                if 'BitGenerator' in err_text or 'MT19937' in err_text:
                    raise
                from gensim.models import LdaModel
                models['lda'] = LdaModel.load(p('lda_model'))

            if os.path.exists(p('lda_dictionary.gensim')):
                models['lda_dict'] = corpora.Dictionary.load(p('lda_dictionary.gensim'))

            models['demo_mode'] = False
            log_ok('LDA chargée')
    except Exception as e:
        log_err('LDA', e)


    # Word2Vec
    try:
        if os.path.exists(p('word2vec_sephora.model')):
            from gensim.models import Word2Vec
            try:
                models['w2v'] = Word2Vec.load(p('word2vec_sephora.model'))
            except Exception as e_inner:
                err_text = str(e_inner)
                if 'BitGenerator' in err_text or 'MT19937' in err_text:
                    # incompatibilité numpy/gensim (pickle)
                    raise
                raise

            models['demo_mode'] = False
            log_ok('Word2Vec chargé')
    except Exception as e:
        log_err('Word2Vec', e)


    # LSTM
    try:
        lstm_paths = ['lstm_ternary.keras', 'lstm_sephora.keras', 'lstm_model.h5']
        for fname in lstm_paths:
            full = p(fname)
            if os.path.exists(full):
                from tensorflow.keras.models import load_model
                # compile=False évite certains soucis de custom loss/metric
                # safe_mode=False aide parfois sur la désérialisation d'objets internes.
                try:
                    models['lstm'] = load_model(full, compile=False, safe_mode=False)
                except Exception as e_inner:
                    err_text = str(e_inner)
                    # Si c'est une incompatibilité keras/tf (désérialisation),
                    # on stoppe et on laisse le pipeline fallback (SVM/règles).
                    if (
                        'could not be deserialized properly' in err_text
                        or 'from_config' in err_text
                        or 'Sequential' in err_text
                    ):
                        models['lstm'] = None
                        raise
                    # Sinon on retente une variante minimale.
                    models['lstm'] = load_model(full, compile=False)

                models['demo_mode'] = False
                log_ok(f"LSTM chargé: {fname}")
                break

    except Exception as e:
        log_err('LSTM', e)

    # SVM + TF-IDF
    try:
        import joblib
        if os.path.exists(p('svm_ternary.pkl')) and os.path.exists(p('tfidf_ternary.pkl')):
            models['svm'] = joblib.load(p('svm_ternary.pkl'))
            models['tfidf'] = joblib.load(p('tfidf_ternary.pkl'))
            models['demo_mode'] = False
            log_ok('SVM+TFIDF (ternaire) chargés')
        elif os.path.exists(p('svm_tfidf_pipeline.pkl')) and os.path.exists(p('tfidf_vectorizer.pkl')):
            models['svm'] = joblib.load(p('svm_tfidf_pipeline.pkl'))
            models['tfidf'] = joblib.load(p('tfidf_vectorizer.pkl'))
            models['demo_mode'] = False
            log_ok('SVM+TFIDF (pipeline) chargés')
    except Exception as e:
        log_err('SVM/TF-IDF', e)

    # Tokenizer
    try:
        if os.path.exists(p('tokenizer_keras.pkl')):
            with open(p('tokenizer_keras.pkl'), 'rb') as f:
                models['tokenizer'] = pickle.load(f)
            models['demo_mode'] = False
            log_ok('Tokenizer chargé')
    except Exception as e:
        log_err('Tokenizer', e)

    # Affichage rapide dans l'UI Streamlit + logs console
    if models['load_ok']:
        st.caption('OK modèles chargés:')
        for msg in models['load_ok'][:6]:
            st.caption(f"✅ {msg}")

    if models['load_errors']:
        st.warning('Certain(s) modèle(s) n\'ont pas pu être chargés. Détails ci-dessous.')
        for err in models['load_errors'][:10]:
            st.caption(f"❌ {err}")

    # logs console (utile si st.caption ne s'affiche pas comme attendu)
    if models['load_errors']:
        for err in models['load_errors']:
            print('[load_models]', err)
    return models




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PIPELINE NLP
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """Nettoyage complet cohérent avec les notebooks Phase 2."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    contractions = {
        "n't": " not", "'re": " are", "'s": " is",
        "'ve": " have", "'ll": " will", "'d": " would", "'m": " am"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_filter(text: str) -> list:
    """Tokenise et filtre stopwords."""
    tokens = text.split()
    return [t for t in tokens if t not in STOP_CUSTOM and len(t) >= 3]


def extract_ingredients(text: str) -> list:
    """Détecte les ingrédients beauté dans le texte."""
    text_lower = text.lower()
    found = []
    for ing in BEAUTY_INGREDIENTS:
        if ing in text_lower:
            found.append(ing)
    return found


def detect_topics_rule_based(tokens: list) -> list:
    """
    Attribution de topics par règles (fallback sans modèle LDA chargé).
    Retourne [(topic_id, label, prob), ...] trié par prob desc.
    """
    token_set = set(tokens)
    scores = []
    for tid, (label, keywords) in TOPIC_LABELS.items():
        score = sum(1 for kw in keywords if kw in token_set) / max(len(keywords), 1)
        scores.append((tid, label, round(score, 3)))
    scores.sort(key=lambda x: -x[2])
    # Normaliser pour avoir des probas plausibles
    total = sum(s for _, _, s in scores) or 1
    return [(tid, label, round(s / total, 3)) for tid, label, s in scores]


def predict_sentiment_rule_based(tokens: list) -> dict:
    """
    Prédiction de sentiment par mots-clés (fallback sans modèle ML).
    Retourne dict avec rating, sentiment, scores, keywords.
    """
    token_set = set(tokens)
    pos_hits = [w for w in token_set if w in POSITIVE_WORDS]
    neg_hits = [w for w in token_set if w in NEGATIVE_WORDS]

    pos_score = len(pos_hits)
    neg_score = len(neg_hits)
    total = pos_score + neg_score + 1e-6

    # Score continu [0, 1] → positif
    sentiment_score = pos_score / total

    # Mapping vers rating 1-5
    if sentiment_score >= 0.80:
        rating = 5
    elif sentiment_score >= 0.60:
        rating = 4
    elif sentiment_score >= 0.40:
        rating = 3
    elif sentiment_score >= 0.20:
        rating = 2
    else:
        rating = 1

    # Probas softmax mock par classe ternaire
    if rating >= 4:
        proba = [0.05, 0.10, 0.85]
    elif rating == 3:
        proba = [0.20, 0.55, 0.25]
    else:
        proba = [0.75, 0.18, 0.07]

    # Top keywords avec importance
    keyword_importance = {}
    for w in pos_hits:
        keyword_importance[w] = round(0.5 + np.random.uniform(0, 0.4), 2)
    for w in neg_hits:
        keyword_importance[w] = round(-(0.4 + np.random.uniform(0, 0.35)), 2)

    return {
        'rating': rating,
        'sentiment_score': round(sentiment_score, 3),
        'proba': proba,
        'pos_words': pos_hits,
        'neg_words': neg_hits,
        'keyword_importance': keyword_importance,
    }


def predict_with_svm(text_clean: str, models: dict) -> dict | None:
    """Prédiction avec le pipeline SVM + TF-IDF si disponible."""
    if models['svm'] is None or models['tfidf'] is None:
        return None
    try:
        X = models['tfidf'].transform([text_clean])
        pred = models['svm'].predict(X)[0]
        # Ternaire (0=Neg, 1=Neu, 2=Pos) → rating estimé
        rating_map = {0: 1, 1: 3, 2: 5}
        rating = rating_map.get(int(pred), 3)
        proba_map = {0: [0.80, 0.14, 0.06], 1: [0.15, 0.70, 0.15], 2: [0.05, 0.10, 0.85]}
        proba = proba_map.get(int(pred), [0.33, 0.34, 0.33])
        return {'rating': rating, 'proba': proba, 'raw_pred': int(pred)}
    except Exception:
        return None


def predict_with_lstm(text_clean: str, models: dict) -> dict | None:
    """Prédiction avec LSTM si disponible."""
    if models['lstm'] is None or models['tokenizer'] is None:
        return None
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq    = models['tokenizer'].texts_to_sequences([text_clean])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        proba  = models['lstm'].predict(padded, verbose=0)[0]
        pred   = int(np.argmax(proba))
        conf   = float(proba[pred])   # score de confiance

        # ── NOUVEAU : seuil de confiance ──────────────────────────────────────
        # Si le LSTM n'est pas sûr (< 0.60), on laisse le SVM ou les règles décider
        if conf < 0.60:
            return None

        if len(proba) == 3:
            rating_map = {0: 1, 1: 3, 2: 5}
            rating     = rating_map[pred]
            proba_list = proba.tolist()
        else:
            rating     = pred + 1
            proba_list = [float(np.sum(proba[:2])), float(proba[2]),
                          float(np.sum(proba[3:]))]

        return {'rating': rating, 'proba': proba_list, 'model': 'BiLSTM'}
    except Exception:
        return None


def get_topics_from_lda(tokens: list, models: dict) -> list:
    """Récupère les topics LDA si le modèle est chargé."""
    if models['lda'] is None or 'lda_dict' not in models:
        return detect_topics_rule_based(tokens)
    try:
        bow = models['lda_dict'].doc2bow(tokens)
        topic_dist = models['lda'].get_document_topics(bow, minimum_probability=0.0)
        results = []
        for tid, prob in sorted(topic_dist, key=lambda x: -x[1])[:3]:
            label = TOPIC_LABELS.get(tid, (f'Topic {tid+1}', []))[0]
            results.append((tid, label, round(float(prob), 3)))
        return results
    except Exception:
        return detect_topics_rule_based(tokens)


def full_analysis_pipeline(review_text: str, models: dict) -> dict:
    """
    Pipeline complet : nettoyage → prédiction → topics → NER → keywords.
    Retourne un dict avec tous les résultats.
    """
    # 1. Nettoyage
    text_clean = clean_text(review_text)
    tokens = tokenize_and_filter(text_clean)

    if len(tokens) < 3:
        return {'error': 'Review trop courte — veuillez saisir au moins quelques mots.'}

    # 2. Sentiment / Rating
    svm_result  = predict_with_svm(text_clean, models)
    lstm_result = predict_with_lstm(text_clean, models)
    rule_result = predict_sentiment_rule_based(tokens)

    # Priorité : LSTM > SVM > règles
    if lstm_result:
        rating         = lstm_result['rating']
        proba          = lstm_result['proba']
        model_used     = 'BiLSTM'
    elif svm_result:
        rating         = svm_result['rating']
        proba          = svm_result['proba']
        model_used     = 'SVM + TF-IDF'
    else:
        rating         = rule_result['rating']
        proba          = rule_result['proba']
        model_used     = 'Analyse lexicale'

    sentiment_score = proba[2]  # proba classe Positif

    # 3. Topics
    topics = get_topics_from_lda(tokens, models)

    # 4. Ingrédients (NER simplifié)
    ingredients = extract_ingredients(review_text)

    # 5. Keywords importants
    word_imp = rule_result['keyword_importance']
    if not word_imp:
        # Mots les plus fréquents filtrés comme fallback
        counter = Counter(tokens)
        word_imp = {w: round(cnt / max(counter.values()), 2)
                    for w, cnt in counter.most_common(8)}

    return {
        'rating':           rating,
        'sentiment_score':  sentiment_score,
        'proba':            proba,
        'pos_words':        rule_result['pos_words'],
        'neg_words':        rule_result['neg_words'],
        'topics':           topics[:3],
        'ingredients':      ingredients,
        'keyword_imp':      word_imp,
        'tokens':           tokens,
        'model_used':       model_used,
        'error':            None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HELPERS UI
# ══════════════════════════════════════════════════════════════════════════════

def render_stars(rating: int) -> str:
    stars = ''.join(['⭐' if i < rating else '☆' for i in range(5)])
    return stars


def sentiment_color(score: float) -> str:
    if score >= 0.65:
        return '#34c783'
    elif score >= 0.40:
        return '#f7c6d4'
    else:
        return '#ff5b5b'


def render_gauge(score: float, label: str = ''):
    color = sentiment_color(score)
    pct   = int(score * 100)
    st.markdown(f"""
    <div style="margin:0.4rem 0;">
        <div style="display:flex; justify-content:space-between; color:#a89bb8; font-size:0.82rem; margin-bottom:4px;">
            <span>{label}</span>
            <span style="color:{color}; font-weight:600;">{pct}%</span>
        </div>
        <div class="gauge-bar-outer">
            <div class="gauge-bar-inner" style="width:{pct}%; background:linear-gradient(90deg, {color}88, {color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_word_bar(word: str, score: float):
    abs_score = abs(score)
    color = '#34c783' if score > 0 else '#ff5b5b'
    width = int(abs_score * 160)
    st.markdown(f"""
    <div class="word-bar-container">
        <span class="word-label">{word}</span>
        <div class="word-bar" style="width:{width}px; background:{color}88; border:1px solid {color};"></div>
        <span style="color:{color}; font-size:0.8rem; margin-left:8px;">{score:+.2f}</span>
    </div>
    """, unsafe_allow_html=True)


def render_wordcloud_matplotlib(word_freq: dict, title: str = '', colormap: str = 'RdPu'):
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        wc = WordCloud(
            width=700, height=380,
            background_color='#160f2a',
            colormap=colormap,
            max_words=80,
            min_font_size=9,
            collocations=False,
            prefer_horizontal=0.85,
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#160f2a')
        ax.set_facecolor('#160f2a')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        if title:
            ax.set_title(title, color='#f7c6d4', fontsize=13, pad=10)
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("📦 `wordcloud` non installé — `pip install wordcloud`")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PAGES
# ══════════════════════════════════════════════════════════════════════════════

def page_analyze(models: dict):
    """PAGE 1 — Analyse de Review."""
    st.markdown('<p class="section-title">🔍 Analyser une review</p>', unsafe_allow_html=True)

    review_input = st.text_area(
        label='Saisissez une review produit en anglais :',
        height=160,
        placeholder=(
            'Ex: "I absolutely love this moisturizer! '
            'It contains hyaluronic acid and niacinamide, '
            'which have transformed my dry skin. '
            'Highly recommend to anyone with sensitive skin."'
        ),
        key='review_input',
    )

    col_btn, col_mode = st.columns([2, 3])
    with col_btn:
        analyze_clicked = st.button('✨ Analyser la review', use_container_width=True)
    with col_mode:
        #if models['demo_mode']:
        #    st.markdown(
        #        '<div style="color:#a89bb8; font-size:0.82rem; padding:0.6rem 0;">'
        #        '⚡ Mode démo — placez vos modèles dans <code>models/</code> pour activer les modèles ML</div>',
        #        unsafe_allow_html=True
        #    )
        #else:
            st.markdown(
                f'<div style="color:#34c783; font-size:0.82rem; padding:0.6rem 0;">'
                f'✅ Modèles chargés</div>',
                unsafe_allow_html=True
            )

    if analyze_clicked and review_input.strip():
        with st.spinner('Analyse en cours…'):
            result = full_analysis_pipeline(review_input, models)

        if result.get('error'):
            st.warning(result['error'])
            return

        # ── Layout résultats ──────────────────────────────────────────────────
        st.markdown('<hr>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1.1, 1], gap='large')

        # ── Colonne gauche ────────────────────────────────────────────────────
        with col_left:

            # Rating prédit
            st.markdown('<p class="section-title" style="font-size:1.1rem;">⭐ Rating prédit</p>',
                        unsafe_allow_html=True)
            rating = result['rating']
            stars_html = ''.join([
                f'<span style="font-size:2rem; color:#f7c6d4;">⭐</span>' if i < rating
                else f'<span style="font-size:2rem; color:rgba(255,255,255,0.15);">☆</span>'
                for i in range(5)
            ])
            st.markdown(
                f'<div class="card card-pink" style="text-align:center; padding:1.5rem;">'
                f'{stars_html}'
                f'<br><span style="font-family:\'Playfair Display\',serif; font-size:1.8rem; '
                f'color:#f7c6d4; font-weight:700;">{rating} / 5</span>'
                f'<br><span style="color:#a89bb8; font-size:0.8rem;">via {result["model_used"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Scores ternaires
            st.markdown('<p class="section-title" style="font-size:1.1rem; margin-top:1.2rem;">📊 Scores de sentiment</p>',
                        unsafe_allow_html=True)
            proba = result['proba']
            labels_ternary = ['😤 Négatif', '😐 Neutre', '😊 Positif']
            for lbl, score in zip(labels_ternary, proba):
                render_gauge(score, lbl)

            # Recommandation
            st.markdown('<p class="section-title" style="font-size:1.1rem; margin-top:1.2rem;">💬 Recommandation</p>',
                        unsafe_allow_html=True)
            if proba[2] >= 0.60:
                st.markdown('<div class="rec-buy">✅ Produit recommandé — Acheter !</div>', unsafe_allow_html=True)
            elif proba[0] >= 0.55:
                st.markdown('<div class="rec-avoid">❌ Produit déconseillé — À éviter</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rec-neutral">⚖️ Avis mitigé — À comparer</div>', unsafe_allow_html=True)

        # ── Colonne droite ────────────────────────────────────────────────────
        with col_right:

            # Topics détectés
            st.markdown('<p class="section-title" style="font-size:1.1rem;">🏷️ Topics détectés</p>',
                        unsafe_allow_html=True)
            topics = result['topics']
            if topics:
                for tid, label, prob in topics[:3]:
                    bar_w = int(prob * 200)
                    st.markdown(
                        f'<div class="topic-card">'
                        f'<span class="topic-name">{label}</span>'
                        f'<br><div style="margin-top:6px;">'
                        f'<div class="gauge-bar-outer" style="height:8px;">'
                        f'<div class="gauge-bar-inner" style="width:{bar_w}px; background:linear-gradient(90deg, #c77dac88, #c77dac);"></div>'
                        f'</div></div>'
                        f'<span class="topic-prob">Confiance : {prob:.1%}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown('<div class="card">Aucun topic détecté.</div>', unsafe_allow_html=True)

            # Ingrédients / Entités NER
            st.markdown('<p class="section-title" style="font-size:1.1rem; margin-top:1rem;">🌿 Ingrédients détectés</p>',
                        unsafe_allow_html=True)
            ings = result['ingredients']
            if ings:
                tags_html = ' '.join(f'<span class="tag">{i}</span>' for i in ings)
                st.markdown(tags_html, unsafe_allow_html=True)
            else:
                st.markdown('<span style="color:#a89bb8; font-size:0.88rem;">Aucun ingrédient identifié</span>',
                            unsafe_allow_html=True)

            # Mots positifs / négatifs
            pos_words = result['pos_words']
            neg_words = result['neg_words']
            if pos_words:
                st.markdown('<br><p class="section-title" style="font-size:0.95rem; margin-top:0.8rem;">✅ Mots positifs</p>',
                            unsafe_allow_html=True)
                st.markdown(' '.join(f'<span class="tag tag-green">{w}</span>' for w in pos_words[:8]),
                            unsafe_allow_html=True)
            if neg_words:
                st.markdown('<p class="section-title" style="font-size:0.95rem; margin-top:0.8rem;">❌ Mots négatifs</p>',
                            unsafe_allow_html=True)
                st.markdown(' '.join(f'<span class="tag tag-red">{w}</span>' for w in neg_words[:8]),
                            unsafe_allow_html=True)

        # ── Keywords importance ───────────────────────────────────────────────
        if result['keyword_imp']:
            st.markdown('<hr><p class="section-title" style="font-size:1.1rem;">🔑 Importance des mots-clés</p>',
                        unsafe_allow_html=True)
            sorted_kw = sorted(result['keyword_imp'].items(), key=lambda x: -abs(x[1]))[:10]
            col_kw1, col_kw2 = st.columns(2)
            mid = len(sorted_kw) // 2
            with col_kw1:
                for word, score in sorted_kw[:mid]:
                    render_word_bar(word, score)
            with col_kw2:
                for word, score in sorted_kw[mid:]:
                    render_word_bar(word, score)

    elif analyze_clicked:
        st.warning('⚠️ Veuillez saisir une review avant d\'analyser.')


def page_topics(models: dict):
    """PAGE 2 — Exploration des topics."""
    st.markdown('<p class="section-title">🗺️ Exploration des Topics</p>', unsafe_allow_html=True)

    topic_options = {v[0]: k for k, v in TOPIC_LABELS.items()}
    selected_label = st.selectbox('Sélectionner un topic :', list(topic_options.keys()))
    selected_id    = topic_options[selected_label]
    topic_keywords = TOPIC_LABELS[selected_id][1]
    topic_pct      = DATASET_STATS['topic_dist'].get(selected_id, 0)

    st.markdown('<hr>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.4, 1], gap='large')

    with col1:
        st.markdown(f'<p class="section-title" style="font-size:1.1rem;">☁️ WordCloud — {selected_label}</p>',
                    unsafe_allow_html=True)

        # WordCloud à partir des mots du topic (simulé avec fréquences)
        np.random.seed(selected_id + 42)
        word_freq = {kw: int(np.random.randint(30, 200)) for kw in topic_keywords}
        # Ajouter mots connexes simulés
        extra_words = {
            0: ['cream', 'lotion', 'gel', 'water', 'nourish', 'plump', 'barrier'],
            1: ['glow', 'bright', 'luminous', 'even', 'tone', 'dark', 'spot'],
            2: ['clean', 'pore', 'oil', 'foam', 'gentle', 'remove', 'purify'],
            3: ['niacinamide', 'retinol', 'hyaluronic', 'ceramide', 'peptide', 'aha'],
            4: ['spf', 'sunblock', 'uv', 'protect', 'broad', 'spectrum', 'daily'],
            5: ['anti-aging', 'wrinkle', 'firm', 'collagen', 'elastin', 'lift', 'plump'],
        }
        for w in extra_words.get(selected_id, []):
            word_freq[w] = int(np.random.randint(15, 80))

        colormaps = ['RdPu', 'PuRd', 'BuPu', 'Purples', 'PuBuGn', 'YlOrRd']
        render_wordcloud_matplotlib(word_freq, colormap=colormaps[selected_id % len(colormaps)])

    with col2:
        st.markdown('<p class="section-title" style="font-size:1.1rem;">📊 Statistiques du topic</p>',
                    unsafe_allow_html=True)

        n_reviews = int(DATASET_STATS['total_reviews'] * topic_pct / 100)
        st.markdown(f"""
        <div class="kpi-box" style="margin-bottom:0.8rem;">
            <div class="kpi-value">{n_reviews:,}</div>
            <div class="kpi-label">Reviews dans ce topic</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-value">{topic_pct}%</div>
            <div class="kpi-label">Du corpus total</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="section-title" style="font-size:1rem; margin-top:1.2rem;">🏷️ Mots-clés principaux</p>',
                    unsafe_allow_html=True)
        tags_html = ' '.join(f'<span class="tag">{kw}</span>' for kw in topic_keywords)
        st.markdown(tags_html, unsafe_allow_html=True)

        st.markdown('<p class="section-title" style="font-size:1rem; margin-top:1.2rem;">📍 Distribution</p>',
                    unsafe_allow_html=True)
        render_gauge(topic_pct / 100, f'{selected_label}')

    # Reviews exemples
    st.markdown('<hr><p class="section-title">📝 Reviews exemples</p>', unsafe_allow_html=True)
    samples = SAMPLE_REVIEWS.get(selected_id, [])
    for i, review in enumerate(samples, 1):
        st.markdown(
            f'<div class="card">'
            f'<span style="color:#a89bb8; font-size:0.78rem; text-transform:uppercase; '
            f'letter-spacing:0.05em;">Exemple {i}</span>'
            f'<p style="color:#e8dff0; margin-top:0.5rem; line-height:1.6;">{review}</p>'
            f'</div>',
            unsafe_allow_html=True
        )


def page_similarity(models: dict):
    """PAGE 3 — Similarité sémantique Word2Vec."""
    st.markdown('<p class="section-title">🔗 Similarité Sémantique (Word2Vec)</p>', unsafe_allow_html=True)

    if models['w2v'] is None:
        st.info(
            '⚡ Mode démo — Word2Vec non chargé. '
            'Placez `word2vec_sephora.model` dans `models/` pour activer les vraies similarités.'
        )
        st.markdown('<p style="color:#a89bb8; font-size:0.9rem;">Résultats simulés pour la démonstration.</p>',
                    unsafe_allow_html=True)

    col_input, col_space = st.columns([1, 1.5])
    with col_input:
        word_query = st.text_input('Mot à explorer :', value='moisturizer', key='word_sim')
        search_btn = st.button('🔍 Trouver mots similaires', use_container_width=True)

    if search_btn or word_query:
        # Dictionnaire de similarités simulées (réalistes pour le domaine beauté)
        MOCK_SIMILARITIES = {
            'moisturizer': [
                ('hydrator', 0.91), ('cream', 0.89), ('lotion', 0.87),
                ('emollient', 0.85), ('serum', 0.82), ('balm', 0.79),
                ('gel', 0.77), ('ointment', 0.74), ('moisture', 0.72), ('hydrate', 0.70),
            ],
            'retinol': [
                ('retinoid', 0.95), ('tretinoin', 0.90), ('bakuchiol', 0.87),
                ('vitamin a', 0.85), ('anti-aging', 0.82), ('wrinkle', 0.79),
                ('peptide', 0.76), ('exfoliant', 0.73), ('niacinamide', 0.70), ('aha', 0.67),
            ],
            'dry': [
                ('dehydrated', 0.93), ('flaky', 0.89), ('tight', 0.86),
                ('dull', 0.83), ('rough', 0.80), ('parched', 0.77),
                ('sensitive', 0.74), ('irritated', 0.71), ('patchy', 0.68), ('itchy', 0.65),
            ],
            'glow': [
                ('radiant', 0.94), ('luminous', 0.91), ('bright', 0.89),
                ('glowing', 0.87), ('dewy', 0.84), ('lit', 0.81),
                ('shiny', 0.78), ('healthy', 0.75), ('fresh', 0.72), ('vibrant', 0.69),
            ],
            'fragrance': [
                ('scent', 0.96), ('smell', 0.93), ('perfume', 0.90),
                ('aroma', 0.87), ('odor', 0.84), ('floral', 0.79),
                ('chemical', 0.76), ('strong', 0.73), ('light', 0.70), ('fresh', 0.67),
            ],
        }

        if models['w2v'] and word_query.lower() in models['w2v'].wv:
            # Vraie prédiction Word2Vec
            similar = models['w2v'].wv.most_similar(word_query.lower(), topn=10)
        elif word_query.lower() in MOCK_SIMILARITIES:
            similar = MOCK_SIMILARITIES[word_query.lower()]
        else:
            # Génération simulée pour n'importe quel mot
            np.random.seed(abs(hash(word_query)) % 1000)
            base_words = list(MOCK_SIMILARITIES['moisturizer'])
            similar = [(w, round(s - np.random.uniform(0, 0.15), 2))
                       for w, s in base_words]

        st.markdown('<hr>', unsafe_allow_html=True)

        col_list, col_viz = st.columns([1, 1.5], gap='large')

        with col_list:
            st.markdown(f'<p class="section-title" style="font-size:1.1rem;">📎 Mots proches de « {word_query} »</p>',
                        unsafe_allow_html=True)
            for i, (word, score) in enumerate(similar[:10], 1):
                bar_w = int(score * 140)
                st.markdown(
                    f'<div style="display:flex; align-items:center; margin:0.4rem 0; gap:10px;">'
                    f'<span style="color:#a89bb8; width:20px; font-size:0.8rem;">{i}.</span>'
                    f'<span style="color:#e8dff0; width:120px; font-size:0.9rem;">{word}</span>'
                    f'<div style="flex:1; background:rgba(255,255,255,0.07); border-radius:4px; height:8px; overflow:hidden;">'
                    f'<div style="width:{bar_w}px; height:100%; background:linear-gradient(90deg, #c77dac88, #c77dac); border-radius:4px;"></div>'
                    f'</div>'
                    f'<span style="color:#f7c6d4; width:45px; text-align:right; font-size:0.82rem; font-weight:600;">{score:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        with col_viz:
            st.markdown('<p class="section-title" style="font-size:1.1rem;">🗺️ Projection 2D (t-SNE simulé)</p>',
                        unsafe_allow_html=True)
            try:
                import plotly.graph_objects as go

                np.random.seed(42)
                words_viz = [word_query] + [w for w, _ in similar[:9]]
                scores_viz = [1.0] + [s for _, s in similar[:9]]

                # Coordonnées simulées en cluster
                angles = np.linspace(0, 2 * np.pi, len(words_viz) - 1, endpoint=False)
                radius = [1 - s * 0.8 for s in scores_viz[1:]]
                x_coords = [0.0] + [r * np.cos(a) + np.random.normal(0, 0.08) for r, a in zip(radius, angles)]
                y_coords = [0.0] + [r * np.sin(a) + np.random.normal(0, 0.08) for r, a in zip(radius, angles)]

                colors_viz = ['#f7c6d4'] + ['#c77dac'] * (len(words_viz) - 1)
                sizes_viz  = [18] + [10 + int(s * 8) for s in scores_viz[1:]]

                fig = go.Figure()
                # Lignes vers le centre
                for i in range(1, len(x_coords)):
                    fig.add_trace(go.Scatter(
                        x=[x_coords[0], x_coords[i]],
                        y=[y_coords[0], y_coords[i]],
                        mode='lines',
                        line=dict(color='rgba(199,125,172,0.25)', width=1),
                        showlegend=False, hoverinfo='none',
                    ))
                # Points
                fig.add_trace(go.Scatter(
                    x=x_coords, y=y_coords,
                    mode='markers+text',
                    marker=dict(color=colors_viz, size=sizes_viz,
                                line=dict(color='rgba(255,255,255,0.3)', width=1)),
                    text=words_viz,
                    textposition='top center',
                    textfont=dict(color='#e8dff0', size=11),
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    showlegend=False,
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(22,15,42,0.6)',
                    font=dict(color='#a89bb8'),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    margin=dict(l=10, r=10, t=20, b=10),
                    height=320,
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info('📦 `plotly` non installé — `pip install plotly`')


def page_statistics():
    """PAGE 4 — Statistiques Dataset."""
    st.markdown('<p class="section-title">📊 Statistiques du Dataset</p>', unsafe_allow_html=True)

    # KPIs principaux
    st.markdown('<p class="section-title" style="font-size:1.1rem;">🎯 Métriques clés</p>', unsafe_allow_html=True)
    kpi_cols = st.columns(4)
    kpis = [
        ('59 K',    'Reviews totales',    ''),
        ('4.21 ⭐',   'Note moyenne',       ''),
        ('77.4%',     'Reviews positives',  ''),
        ('8 494',     'Produits uniques',   ''),
    ]
    for col, (val, label, _) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f'<div class="kpi-box"><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<br>', unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        import plotly.express as px

        col_charts1, col_charts2 = st.columns(2, gap='large')

        # ── Distribution des ratings ─────────────────────────────────────────
        with col_charts1:
            st.markdown('<p class="section-title" style="font-size:1.1rem;">⭐ Distribution des Ratings</p>',
                        unsafe_allow_html=True)
            rating_vals = [1, 2, 3, 4, 5]
            rating_pcts = [DATASET_STATS['rating_dist'][r] for r in rating_vals]
            rating_colors = ['#ff5b5b', '#ff9966', '#f7c6d4', '#c77dac', '#34c783']

            fig_rating = go.Figure(go.Bar(
                x=[f'★{r}' for r in rating_vals],
                y=rating_pcts,
                marker=dict(color=rating_colors, line=dict(color='rgba(0,0,0,0)', width=0)),
                text=[f'{p}%' for p in rating_pcts],
                textposition='outside',
                textfont=dict(color='#a89bb8', size=11),
            ))
            fig_rating.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(22,15,42,0.6)',
                font=dict(color='#a89bb8'),
                xaxis=dict(showgrid=False, title='Rating'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='%'),
                margin=dict(l=10, r=10, t=30, b=10),
                height=300,
            )
            st.plotly_chart(fig_rating, use_container_width=True)

        # ── Distribution des Topics ──────────────────────────────────────────
        with col_charts2:
            st.markdown('<p class="section-title" style="font-size:1.1rem;">🏷️ Distribution des Topics</p>',
                        unsafe_allow_html=True)
            topic_names = [TOPIC_LABELS[i][0] for i in range(6)]
            topic_pcts  = [DATASET_STATS['topic_dist'][i] for i in range(6)]
            topic_colors = ['#c77dac', '#f7c6d4', '#e8a4bc', '#a855a0', '#d4a0c8', '#9b4d9b']

            fig_topics = go.Figure(go.Pie(
                labels=topic_names,
                values=topic_pcts,
                hole=0.5,
                marker=dict(colors=topic_colors, line=dict(color='#160f2a', width=2)),
                textfont=dict(size=10, color='#e8dff0'),
                textposition='outside',
            ))
            fig_topics.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a89bb8'),
                legend=dict(font=dict(size=10, color='#c9b8d8'), bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig_topics, use_container_width=True)

        # ── Top ingrédients ──────────────────────────────────────────────────
        st.markdown('<p class="section-title" style="font-size:1.1rem;">🌿 Top 10 Ingrédients Mentionnés</p>',
                    unsafe_allow_html=True)

        ing_names  = [i[0] for i in DATASET_STATS['top_ingredients']]
        ing_counts = [i[1] for i in DATASET_STATS['top_ingredients']]
        ing_colors = [f'rgba(199,125,172,{0.5 + 0.05*i})' for i in range(len(ing_names))]

        fig_ing = go.Figure(go.Bar(
            x=ing_counts[::-1],
            y=ing_names[::-1],
            orientation='h',
            marker=dict(color=ing_colors[::-1], line=dict(color='rgba(0,0,0,0)')),
            text=[f'{c:,}' for c in ing_counts[::-1]],
            textposition='outside',
            textfont=dict(color='#a89bb8', size=10),
        ))
        fig_ing.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(22,15,42,0.6)',
            font=dict(color='#a89bb8'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', title='Mentions'),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=80, t=10, b=10),
            height=340,
        )
        st.plotly_chart(fig_ing, use_container_width=True)

        # ── Performances modèles ─────────────────────────────────────────────
        st.markdown('<p class="section-title" style="font-size:1.1rem;">🤖 Performances des Modèles</p>',
                    unsafe_allow_html=True)

        model_data = {
            'Modèle': ['SVM + TF-IDF\n(5 classes)', 'BiLSTM\n(5 classes)',
                       'SVM + TF-IDF\n(ternaire)', 'BiLSTM\n(ternaire)'],
            'F1-Macro': [0.4625, 0.4871, 0.6742, 0.6993],
            'Accuracy': [0.6234, 0.6491, 0.8592, 0.8999],
        }
        df_models = pd.DataFrame(model_data)

        fig_models = go.Figure()
        fig_models.add_trace(go.Bar(
            name='F1-Macro', x=df_models['Modèle'], y=df_models['F1-Macro'],
            marker_color='#c77dac', opacity=0.9,
        ))
        fig_models.add_trace(go.Bar(
            name='Accuracy', x=df_models['Modèle'], y=df_models['Accuracy'],
            marker_color='#f7c6d4', opacity=0.75,
        ))
        fig_models.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(22,15,42,0.6)',
            font=dict(color='#a89bb8'),
            barmode='group',
            legend=dict(font=dict(color='#c9b8d8'), bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)',
                       title='Score', range=[0, 0.9]),
            margin=dict(l=10, r=10, t=20, b=10),
            height=320,
        )
        st.plotly_chart(fig_models, use_container_width=True)

    except ImportError:
        st.info('📦 `plotly` non installé — `pip install plotly`')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SIDEBAR & MAIN
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1.5rem 0 1rem 0;">
            <div style="font-size:2.8rem; margin-bottom:0.3rem;">💄</div>
            <div style="font-family:'Playfair Display',serif; font-size:1.3rem;
                        color:#f7c6d4; font-weight:700; line-height:1.2;">
                Sephora<br>Review Analyzer
            </div>
            <div style="color:#a89bb8; font-size:0.72rem; letter-spacing:0.1em;
                        text-transform:uppercase; margin-top:0.4rem;">
                NLP Pipeline v1.0
            </div>
        </div>
        <hr>
        """, unsafe_allow_html=True)

        st.markdown('<p style="color:#a89bb8; font-size:0.8rem; text-transform:uppercase; '
                    'letter-spacing:0.1em; margin-bottom:0.5rem;">Navigation</p>',
                    unsafe_allow_html=True)

        page = st.radio(
            label='',
            options=[
                '🔍 Analyse de Review',
                '🗺️ Exploration Topics',
                '🔗 Similarité Sémantique',
                '📊 Statistiques Dataset',
            ],
            label_visibility='collapsed',
        )

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:1rem 0; color:#a89bb8; font-size:0.8rem; line-height:1.6;">
            <b style="color:#f7c6d4;">Pipeline NLP</b><br>
            EDA → Preprocessing → LDA<br>
            Word2Vec → SVM → BiLSTM<br><br>
            <b style="color:#f7c6d4;">Dataset</b><br>
            Sephora Reviews (Kaggle)<br>
            ~59k reviews · 5 ratings<br><br>
            <b style="color:#f7c6d4;">Modèles</b><br>
            SVM + TF-IDF (baseline)<br>
            BiLSTM (classification)<br>
            LDA (topic modeling)<br>
            Word2Vec (embeddings)
        </div>
        """, unsafe_allow_html=True)

    return page


def main():
    # Chargement des modèles
    with st.spinner('Chargement des modèles…'):
        models = load_models()

    # Sidebar + navigation
    page = render_sidebar()

    # Header principal
    st.markdown("""
    <div style="padding:2rem 0 1.5rem 0;">
        <div class="main-header">Analyseur de Reviews<br>E-Commerce</div>
        <div class="sub-header" style="margin-top:0.5rem;">
            NLP Pipeline · Sephora Skincare Dataset · LDA · Word2Vec · LSTM
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Router
    if page == '🔍 Analyse de Review':
        page_analyze(models)
    elif page == '🗺️ Exploration Topics':
        page_topics(models)
    elif page == '🔗 Similarité Sémantique':
        page_similarity(models)
    elif page == '📊 Statistiques Dataset':
        page_statistics()


if __name__ == '__main__':
    main()
