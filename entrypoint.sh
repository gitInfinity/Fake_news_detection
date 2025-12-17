#!/bin/bash
set -e

# Download required NLTK data if not already present
python - <<PY
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
PY

# Execute the command
exec "$@"
