import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Tuple

import os
import logging
from models import load_w2v_model, load_keras_model

# Default settings
EMBEDDING_DIM = 100

logger = logging.getLogger(__name__)
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    # If stopwords are missing, caller should ensure nltk.download('stopwords','punkt') ran
    STOP_WORDS = set()


def preprocess_text(text: str):
    """Lowercase, remove punctuation, tokenize and remove stop words."""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    if STOP_WORDS:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def get_document_vector(word_list, model, vector_size=EMBEDDING_DIM):
    """Compute average Word2Vec vector for a document."""
    vector = np.zeros(vector_size)
    word_count = 0
    if hasattr(model, 'wv'):
        wv = model.wv
        for w in word_list:
            if w in wv:
                vector += wv[w]
                word_count += 1
    if word_count > 0:
        vector = vector / word_count
    return vector


def load_models() -> Tuple[object, object]:
    """Helper to load both models and return (w2v_model, keras_model).

    Raises FileNotFoundError if models are missing.
    """
    # Prefer models placed in the `models/` directory if present
    default_w2v_path = os.path.join(os.getcwd(), "models", "word2vec_model.model")
    default_keras_path = os.path.join(os.getcwd(), "models", "fake_news_classifier.h5")

    try:
        w2v_path = default_w2v_path if os.path.exists(default_w2v_path) else None
        keras_path = default_keras_path if os.path.exists(default_keras_path) else None

        w2v = load_w2v_model(w2v_path)
        keras = load_keras_model(keras_path)

        # Attempt to compile the model so compiled metrics are available (safe for inference)
        try:
            keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        except Exception:
            # Compilation is optional for inference; ignore failures
            logger.debug("Could not compile Keras model after loading.")

        return w2v, keras
    except FileNotFoundError as e:
        # Reraise with a clearer message for the Streamlit UI
        logger.error("Model loading failed: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error loading models: %s", e)
        raise
