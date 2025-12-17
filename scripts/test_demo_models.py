from model_util import load_models, preprocess_text, EMBEDDING_DIM, get_document_vector

w2v, clf = load_models()
print('w2v ok:', hasattr(w2v, 'wv'))
text = 'President announces new education funding for schools'
toks = preprocess_text(text)
vec = get_document_vector(toks, w2v, EMBEDDING_DIM)
print('vector shape:', vec.shape)
print('vector sample[0:5]:', vec[:5])
pred = clf.predict(vec.reshape(1, -1))
print('prediction (prob):', pred)
