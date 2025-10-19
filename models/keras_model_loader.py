import os
from tensorflow.keras.models import load_model

MODEL_FILENAME = "fake_news_classifier.h5"


def load_keras_model(path=None):
    """Load a Keras model from the given path or from the project root.

    Returns the loaded model or raises FileNotFoundError.
    """
    if path is None:
        path = os.path.join(os.getcwd(), MODEL_FILENAME)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Keras model not found at: {path}")

    return load_model(path)
