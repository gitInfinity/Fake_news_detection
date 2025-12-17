import streamlit as st
import pandas as pd
import numpy as np
import re
from model_util import preprocess_text, get_document_vector, load_models, EMBEDDING_DIM, using_demo_models

st.set_page_config(page_title="Fake News Detector", layout="wide", initial_sidebar_state="collapsed")

word2vec_model, classifier = load_models()

# If demo models are active, show a banner so users know this is a lightweight demo
if using_demo_models():
    st.info("Demo mode: running with lightweight demo models. Replace models in `models/` to use your trained models.")

st.title("📰 AI Fake News Classifier")
st.markdown("Enter a news headline or article text below. The model will analyze its content and predict whether it is **True** or **Fake**.")
st.divider()

user_input = st.text_area("Paste News Article Text Here:", height=250, 
                          placeholder="e.g., 'The President has signed an order banning all use of social media.'")

if st.button("Analyze Article", type="primary"):
    
    if not user_input or classifier is None:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing content...'):
            
            tokenized_text = preprocess_text(user_input)
            
            if not tokenized_text:
                st.error("The text provided is too short or contains no recognizable words after cleaning. Please try again with a longer article.")
            else:
                doc_vector = get_document_vector(tokenized_text, word2vec_model, EMBEDDING_DIM)
                
                X_predict = doc_vector.reshape(1, EMBEDDING_DIM)
                
                prediction_proba = classifier.predict(X_predict, verbose=0)[0][0]
                
                if prediction_proba >= 0.5:
                    result_label = "TRUE NEWS"
                else:
                    result_label = "FAKE NEWS"
                
                st.divider()
                st.subheader("Analysis Result")

                col1, col2, col3 = st.columns([1, 1.5, 1])

                with col2:
                    st.metric(
                        label="Predicted Classification",
                        value=result_label,
                        delta=f"Confidence Score: {prediction_proba * 100:.2f}%",
                        delta_color="off"
                    )

                if result_label == "TRUE NEWS":
                    st.success(f"✅ This article appears to be **{result_label}**.")
                else:
                    st.error(f"❌ This article appears to be **{result_label}**.")

                with st.expander("See technical details"):
                    st.write(f"Raw Prediction Score (0=Fake, 1=True): `{prediction_proba:.4f}`")
                    st.write(f"Input Vector Shape: `{X_predict.shape}`")
                    st.write(f"Tokens found after preprocessing: `{len(tokenized_text)}`")