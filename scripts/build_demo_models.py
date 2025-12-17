"""Build tiny demo Word2Vec and scikit-learn classifier for local demos.
Saves:
 - models/word2vec_demo.model
 - models/demo_classifier.joblib

This script is intentionally small and deterministic so it runs fast inside CI or a dev machine.
"""
import os
from pathlib import Path
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import joblib

# Simple demo corpus and labels
DOCS = [
    ("government announces new tax relief to support families", 1),
    ("scientists discover cure for common cold", 1),
    ("local man wins lottery twice in one day", 0),
    ("celebrity caught in scandal that never happened", 0),
    ("city invests in renewable energy projects", 1),
    ("miracle weight loss pill approved by doctors", 0),
    ("school receives grant for new computers", 1),
    ("fake rumor spreads about mayor", 0),
]

EMBEDDING_DIM = 100

def simple_tokenize(text):
    return [w.strip('.,!?:;').lower() for w in text.split() if w.strip('.,!?:;')]

def build_word2vec(sentences, vector_size=EMBEDDING_DIM):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=5, min_count=1, workers=1, sg=1)
    return model

def doc_to_vector(tokens, w2v, dim=EMBEDDING_DIM):
    vec = np.zeros(dim)
    count = 0
    for t in tokens:
        if t in w2v.wv:
            vec += w2v.wv[t]
            count += 1
    if count > 0:
        vec = vec / count
    return vec


def main():
    out_dir = Path(__file__).resolve().parents[1] / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)

    sentences = [simple_tokenize(text) for text, _ in DOCS]
    labels = np.array([lab for _, lab in DOCS])

    print('Training demo Word2Vec...')
    w2v = build_word2vec(sentences)
    w2v_path = out_dir / 'word2vec_demo.model'
    w2v.save(str(w2v_path))
    print('Saved demo Word2Vec to', w2v_path)

    print('Building document vectors...')
    X = np.vstack([doc_to_vector(s, w2v) for s in sentences])

    print('Training demo classifier (LogisticRegression)...')
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X, labels)

    clf_path = out_dir / 'demo_classifier.joblib'
    joblib.dump(clf, str(clf_path))
    print('Saved demo classifier to', clf_path)

    print('Demo models built successfully.')

if __name__ == '__main__':
    main()
