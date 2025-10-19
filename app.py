import streamlit as st
import pandas as pd
import numpy as np
import re
from model_util import preprocess_text, get_document_vector, load_models, EMBEDDING_DIM

# --- 1. CONFIGURATION & MODEL LOADING ---

st.set_page_config(page_title="Fake News Detector", layout="centered")

# Load models using the shared utility (cached inside the utility if desired)
word2vec_model, classifier = load_models()

# --- 2. PREPROCESSING FUNCTIONS ---

# Note: preprocessing and vector utilities are provided by `model_util`.

# --- 3. STREAMLIT APP LAYOUT ---

st.title("📰 AI Fake News Classifier")
st.markdown("Enter a news headline or article text below to determine if it is likely **Real** (1) or **Fake** (0).")
st.markdown("---")

# Text input box for the user
user_input = st.text_area("Paste News Article Text Here:", height=200, 
                          placeholder="e.g., 'The President has signed an order banning all use of social media.'")

# Prediction button
if st.button("Analyze Article", type="primary"):
    
    if not user_input or classifier is None:
        st.warning("Please enter some text and ensure models are loaded correctly.")
    else:
        # --- 4. PREDICTION PIPELINE ---
        with st.spinner('Analyzing content...'):
            
            # 1. Preprocess the input text
            tokenized_text = preprocess_text(user_input)
            
            if not tokenized_text:
                st.error("The text provided is too short or contains no recognizable words after cleaning.")
            else:
                # 2. Generate the document vector
                doc_vector = get_document_vector(tokenized_text, word2vec_model, EMBEDDING_DIM)
                
                # Reshape the vector for the Keras model (1 sample, 100 features)
                X_predict = doc_vector.reshape(1, EMBEDDING_DIM)
                
                # 3. Make the prediction (outputs a probability)
                prediction_proba = classifier.predict(X_predict, verbose=0)[0][0]
                
                # 4. Convert probability to class
                if prediction_proba >= 0.5:
                    result_label = "TRUE NEWS"
                else:
                    result_label = "FAKE NEWS"
                
                # --- 5. DISPLAY RESULTS ---
                st.markdown("### Classification Result:")
                
                # Use st.metric with the CORRECT delta_color value ("off")
                # and use delta for the descriptive prediction.
                st.metric(
                    label="Model Confidence Score", 
                    value=f"{prediction_proba * 100:.2f}%", 
                    delta=f"Predicted as {result_label}",
                    delta_color="off" # FIX APPLIED: Uses 'off' instead of 'success'/'error'
                )
                
                # Provide clear, colored feedback outside of the metric
                if result_label == "TRUE NEWS":
                    st.success(f"✅ Prediction: This article is likely **{result_label}**.")
                else:
                    st.error(f"❌ Prediction: This article is likely **{result_label}**.")