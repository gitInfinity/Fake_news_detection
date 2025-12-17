import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Tuple

import os
import logging
from models import load_w2v_model, load_keras_model
import joblib


class SklearnClassifierWrapper:
    """Wrap a scikit-learn classifier to provide a Keras-like `predict` API

    `predict(X, verbose=0)` returns an array of shape (n_samples, 1) with
    the positive-class probability in the single column, matching how the
    original Keras model was used in the app.
    """
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X, verbose=0):
        # ensure numpy array
        import numpy as _np
        probs = self.clf.predict_proba(_np.asarray(X))[:, 1]
        return probs.reshape(-1, 1)

# Default settings
EMBEDDING_DIM = 100

logger = logging.getLogger(__name__)
DEMO_MODE = False
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

    global DEMO_MODE

    try:
        # prefer real model names
        w2v_path = default_w2v_path if os.path.exists(default_w2v_path) else None
        keras_path = default_keras_path if os.path.exists(default_keras_path) else None

        # fallback demo model paths
        demo_w2v_path = os.path.join(os.getcwd(), "models", "word2vec_demo.model")
        demo_clf_path = os.path.join(os.getcwd(), "models", "demo_classifier.joblib")

        # Load Word2Vec: prefer real model, otherwise load demo if present
        if w2v_path is not None:
            w2v = load_w2v_model(w2v_path)
        elif os.path.exists(demo_w2v_path):
            w2v = load_w2v_model(demo_w2v_path)
            logger.info("Loaded demo Word2Vec model")
            DEMO_MODE = True
        else:
            # try default loader with None (it will raise FileNotFoundError)
            w2v = load_w2v_model(w2v_path)

        # Load classifier: prefer Keras .h5; if not present, fall back to sklearn demo
        if keras_path is not None:
            keras = load_keras_model(keras_path)
            # Try to compile so metrics are available (optional)
            try:
                keras.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            except Exception:
                logger.debug("Could not compile Keras model after loading.")
        elif os.path.exists(demo_clf_path):
            clf = joblib.load(demo_clf_path)
            keras = SklearnClassifierWrapper(clf)
            logger.info("Loaded demo scikit-learn classifier and wrapped for predict().")
            DEMO_MODE = True
        else:
            # load_keras_model will raise FileNotFoundError if no model found
            keras = load_keras_model(keras_path)

        return w2v, keras
    except FileNotFoundError as e:
        # Reraise with a clearer message for the Streamlit UI
        logger.error("Model loading failed: %s", e)
        raise
    except Exception as e:
        logger.exception("Unexpected error loading models: %s", e)
        raise


def using_demo_models() -> bool:
    """Return True if the demo fallback models were used on load."""
    return DEMO_MODE
