# Fake News Detection

A small end-to-end project that trains and serves a fake-news classifier using a Word2Vec embedding + Keras classifier. The repository includes utilities for preprocessing, model loading, and a Streamlit app for interactive predictions.

This README explains how the project is organized, how to set up a reproducible environment on Windows (PowerShell), how to run the Jupyter notebook and Streamlit app, where to put model files, and how to troubleshoot common issues (TensorFlow native DLL errors, missing NLTK data, gensim import errors, etc.).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Requirements & Supported Platforms](#requirements--supported-platforms)
- [Quickstart (Windows PowerShell)](#quickstart-windows-powershell)
- [Create and activate a virtual environment](#create-and-activate-a-virtual-environment)
- [Install Python packages](#install-python-packages)
- [Download NLTK data](#download-nltk-data)
- [Jupyter Notebook (exploration & training)](#jupyter-notebook-exploration--training)
- [Streamlit app (serving)](#streamlit-app-serving)
- [Model files: placement and loaders](#model-files-placement-and-loaders)
- [Utility API (`model_util.py`)](#utility-api-model_utilpy)
- [Training tips: Word2Vec and Keras outline](#training-tips-word2vec-and-keras-outline)
- [Troubleshooting](#troubleshooting)
- [Development & Testing](#development--testing)
- [Git & .gitignore notes](#git--gitignore-notes)

---

## Project Overview

This project demonstrates a simple architecture for fake-news detection:

1. Text preprocessing with NLTK tokenization + stopword removal
2. Train a Word2Vec model on the tokenized corpus to create fixed-length embeddings
3. Convert each document to an average Word2Vec vector
4. Train a Keras (TensorFlow) binary classifier on document vectors
5. Serve predictions via a Streamlit web UI

The repository also contains convenience loader modules and a `model_util.py` helper that centralizes the preprocessing, vectorization, and model-loading logic used by the Streamlit app.

---

## Repository Structure

Example top-level layout (your workspace may contain additional files):

```
app.py                        # Streamlit application
model_util.py                 # Shared utilities (preprocess, vectorize, load_models)
main.ipynb                    # Notebook used for training/experimentation
fake_news_classifier.h5       # Trained Keras model (NOT checked in; add to .gitignore)
word2vec_model.model          # Trained gensim Word2Vec model (NOT checked in)
requirements.txt              # Python requirements (optional)
.gitignore
/models/                      # package - loaders and optional model storage
    __init__.py
    w2v_model_loader.py
    keras_model_loader.py
    README.md
README.md                     # This file
.venv/                        # recommended virtual environment (ignored)
```

---

## Requirements & Supported Platforms

- OS: Windows 10/11 (instructions use PowerShell)
- Python: 3.10 or 3.11 recommended (64-bit required)
- TensorFlow: CPU or GPU wheel compatible with your Python version (this repo uses TensorFlow via pip)
- Packages: streamlit, gensim, nltk, scikit-learn, tensorflow, numpy, pandas, keras, etc.

Note: TensorFlow has native platform-specific binaries. The most common source of runtime ImportError/DLL load failures is an incompatible Python/TensorFlow wheel, missing Visual C++ Redistributable, or incorrect CUDA/cuDNN for GPU installations. See the Troubleshooting section.

---

## Quickstart (Windows PowerShell)

Open a PowerShell terminal in the project root. The following steps create a virtual environment, install packages, download NLTK data, and run the Streamlit app.

### Create and activate a virtual environment

```powershell
# Create (requires a compatible Python on PATH, e.g., py -3.11 if installed)
py -3.11 -m venv .venv

# Activate in PowerShell
. .\.venv\Scripts\Activate.ps1

# Confirm
python --version
```

If you used `py -3.11` and received an error, install a compatible Python (3.11 or 3.10) and retry. You can install Python from https://www.python.org/downloads/windows/ or using `winget`.

### Install Python packages

```powershell
# upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# install project dependencies
python -m pip install streamlit gensim nltk scikit-learn tensorflow ipykernel
```

Note on TensorFlow: If you encounter a DLL import failure (see Troubleshooting), try installing a specific compatible TensorFlow version such as `tensorflow==2.14.1` (stable for Python 3.11) or follow the TensorFlow install matrix for GPU support.

### Download NLTK data

Run the following so tokenizers and stopwords are available to the notebook and app:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Register the virtualenv as a Jupyter kernel (optional)

```powershell
python -m ipykernel install --user --name fake-news-venv --display-name "Python (fake-news)"
```

---

## Jupyter Notebook (exploration & training)

Open `main.ipynb` for exploratory data analysis and model training steps (corpus preprocessing, Word2Vec training, vectorization, classifier training). Use the virtual environment or the kernel you registered above.

In general, the notebook covers:
- Loading CSVs (True.csv, Fake.csv)
- Preprocessing (tokenize, lower, clean punctuation, remove stopwords)
- Train gensim Word2Vec (configurable vector_size/window/min_count)
- Save Word2Vec model (`word2vec_model.model`)
- Create document vectors (average of known Word2Vec vectors per document)
- Train Keras model and save as `fake_news_classifier.h5`

If you need to re-run the notebook headless or programmatically, use:

```powershell
jupyter notebook
# or
jupyter lab
```

---

## Streamlit app (serving)

Run the Streamlit app to interactively test articles:

```powershell
# Make sure .venv is activated
streamlit run app.py
```

Behavior:
- App loads the Word2Vec model and Keras model at startup (via `model_util.load_models()`)
- User pastes article text in the text area and clicks `Analyze Article`
- Preprocessing -> get_document_vector -> classifier.predict -> display result

If models cannot be found, the app shows an error message with guidance.

---

## Model files: placement and loaders

Model binary files are intentionally ignored by `.gitignore` because they are large. Place your trained files in either the project root or the `models/` folder. Loaders will prefer `models/` when present.

Default file names expected (change the loaders if you prefer different names):

- `models/word2vec_model.model` or `word2vec_model.model`
- `models/fake_news_classifier.h5` or `fake_news_classifier.h5`

The `models` package contains convenience loaders:
- `models/w2v_model_loader.py` -> `load_w2v_model(path=None)`
- `models/keras_model_loader.py` -> `load_keras_model(path=None)`
- `models.__init__.py` exposes `load_models(w2v_path=None, keras_path=None)`

`model_util.load_models()` delegates to these loaders and attempts to compile the Keras model (wrapped in try/except) to populate compiled metrics when possible.

---

## Utility API (`model_util.py`)

The `model_util.py` module centralizes preprocessing and model-loading utilities used by the Streamlit app.

Public functions and constants:

- `EMBEDDING_DIM` (int) - expected Word2Vec vector size used by the app (default: 100)
- `preprocess_text(text: str) -> list[str]` - lowercases, removes punctuation, tokenizes, removes stopwords
- `get_document_vector(word_list, model, vector_size=EMBEDDING_DIM) -> numpy.ndarray` - returns average Word2Vec vector for the document
- `load_models() -> (w2v_model, keras_model)` - loads both models (prefers `models/` directory if present); attempts to compile the Keras model after load

The Streamlit app imports these functions directly:

```python
from model_util import preprocess_text, get_document_vector, load_models, EMBEDDING_DIM
```

---

## Training tips: Word2Vec and Keras outline

### Quick Word2Vec recipe (gensim)

```python
from gensim.models import Word2Vec

sentences = df['processed_text'].tolist()
model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # EMBEDDING_DIM
    window=5,
    min_count=5,
    sg=1,               # skip-gram (1) or cbow (0)
    workers=4
)
model.save('word2vec_model.model')
```

### Quick Keras binary classifier skeleton

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential([
    Dense(128, activation='relu', input_dim=EMBEDDING_DIM),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)
model.save('fake_news_classifier.h5')
```

Notes:
- Always ensure `EMBEDDING_DIM` used in model training matches the one used in the app.
- Save both the Word2Vec and Keras model files and place them under `models/` for production-style serving.

---

## Troubleshooting

This section collects common issues and actionable fixes.

### 1) ImportError: No module named 'gensim'

Install gensim in the same Python environment that runs the notebook or Streamlit app:

```powershell
python -m pip install gensim
```

If your editor still flags the import after installation, ensure VS Code's selected interpreter matches your venv.

### 2) NLTK LookupError - stopwords or punkt not found

Run this once in the same environment your notebook/Streamlit uses:

```powershell
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

Restart the kernel or Streamlit if you still see errors.

### 3) Failed to load the native TensorFlow runtime / DLL load failed

This is the most common runtime issue on Windows. Steps to diagnose and fix:

1. Verify Python is 64-bit and a supported version (3.10 or 3.11 recommended)

```powershell
python --version
python -c "import platform; print(platform.architecture())"
```

2. Check the installed TensorFlow wheel/version

```powershell
python -m pip show tensorflow
```

3. If the import fails with the DLL error, try:

- Install the Microsoft Visual C++ Redistributable (x64):
  https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
- Reinstall a stable TensorFlow CPU wheel for your Python version, for example:

```powershell
python -m pip uninstall -y tensorflow keras tensorflow-estimator
python -m pip install tensorflow==2.14.1
```

4. If you want GPU support, follow the official TensorFlow GPU compatibility matrix and install the exact CUDA and cuDNN versions required by the TensorFlow release.

5. After fixes, test import:

```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

If issues persist, collect the full traceback (see `tf_check.py` technique below) and search/ask on TensorFlow issues.

#### Collect a full import traceback (debug)

Create a small script `tf_check.py` containing:

```python
import traceback, sys
try:
    import tensorflow as tf
    print('ok', tf.__version__)
except Exception:
    traceback.print_exc()
    sys.exit(1)
```

Run it and paste the full traceback in an issue if you need external help.

### 4) Warnings about compiled model metrics or deprecation messages

- `Compiled the loaded model, but the compiled metrics have yet to be built.` — harmless for inference. To populate metric objects, compile the model after loading with the same loss/metrics used during training. `model_util.load_models()` already attempts to `compile()` in a safe try/except.
- Deprecation messages (e.g. `tf.reset_default_graph`) usually come from internals. You can silence TF log noise by setting:

```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # hide INFO and WARNING messages
```

Note: silencing logs only hides messages; it does not fix runtime problems.

---

## Development & Testing

- Add unit tests for `get_document_vector` and `preprocess_text` if you want CI coverage.
- Fast checks you can run locally:

```powershell
# run a quick sanity script that loads models and predicts a dummy example
python - <<'PY'
from model_util import load_models, preprocess_text, get_document_vector, EMBEDDING_DIM
w2v, clf = load_models()
example = "This is a test news article about technology and politics."
tokens = preprocess_text(example)
vec = get_document_vector(tokens, w2v, EMBEDDING_DIM)
print('vector shape', vec.shape)
print('predict (if classifier present):', clf.predict(vec.reshape(1, -1)))
PY
```

---

## Git & .gitignore notes

A `.gitignore` file is included and excludes:
- `.venv/` virtual environment
- compiled python files and caches (`__pycache__`, `*.pyc`)
- model binaries like `*.h5` and `*.model`
- Jupyter checkpoints

This keeps the repository lightweight. If you need to version large model files, prefer an artifact storage or Git LFS.

---

## Optional: TF-free fallback (development convenience)

If you have persistent issues with TensorFlow on Windows and just want to iterate on features quickly, you can temporarily use a scikit-learn MLP classifier (no native TensorFlow DLLs required). Example snippet to use inside `app.py` or the notebook:

```python
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu', max_iter=100, random_state=42)
clf.fit(X_train, Y_train)
```

This is a stop-gap to continue development; model performance may differ from a tuned Keras model.
