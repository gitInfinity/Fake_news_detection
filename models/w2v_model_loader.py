from gensim.models import Word2Vec
import os

MODEL_FILENAME = "word2vec_model.model"


def load_w2v_model(path=None):
    """Load Word2Vec model from the given path or from the project root.

    Returns the loaded model or raises FileNotFoundError.
    """
    if path is None:
        path = os.path.join(os.getcwd(), MODEL_FILENAME)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Word2Vec model not found at: {path}")

    return Word2Vec.load(path)
