"""models package

This package provides helpers to load the project's trained models.

The file was initially created as a package marker. I've added a few
convenience imports so you can do::

	from models import load_w2v_model, load_keras_model, load_models

or::

	import models
	w2v, keras = models.load_models()

If you prefer to keep model loading logic centralized in `model_util.py`,
you can still use that module; these helpers are just convenience wrappers.
"""

from .w2v_model_loader import load_w2v_model
from .keras_model_loader import load_keras_model


def load_models(w2v_path=None, keras_path=None):
	"""Load Word2Vec and Keras models and return (w2v_model, keras_model).

	Parameters:
		w2v_path: optional explicit filesystem path to the Word2Vec model.
		keras_path: optional explicit filesystem path to the Keras .h5 model.
	"""
	w2v = load_w2v_model(w2v_path)
	keras = load_keras_model(keras_path)
	return w2v, keras


__all__ = ["load_w2v_model", "load_keras_model", "load_models"]
