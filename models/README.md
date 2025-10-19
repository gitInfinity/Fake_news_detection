Models directory

Place your trained model files here (or in the project root):

- word2vec_model.model   -> Gensim Word2Vec model
- fake_news_classifier.h5 -> Keras .h5 model

The loaders in `models/w2v_model_loader.py` and `models/keras_model_loader.py`
look for files in the project root by default. You can pass an explicit path
when calling the loader functions if you store the files elsewhere.