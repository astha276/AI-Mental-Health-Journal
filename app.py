import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# --- NEW: Import LIME and related libraries ---
import lime
import lime.lime_text
from sklearn.pipeline import make_pipeline

# --- Load the Trained Model ---
try:
    model = joblib.load('mental_health_model.pkl')
except FileNotFoundError:
    st.error("Model file not found. Please run `train_model.py` first to create the model.")
    st.stop()

# --- Helper Function for Cleaning Text ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- NEW: Create a LIME Explainer ---
# LIME needs a function that takes text and returns prediction probabilities.
# Our scikit-learn pipeline does this, so we can wrap it.
def predictor(texts):
    cleaned_texts = [clean_text(text) for text in texts]
    return model.predict_proba(cleaned_texts)

# Create the LIME text explainer
explainer = lime.lime_text.LimeTextExplainer(class_names=['Neutral', 'Distress'])

# --- Initialize Session State for Storing Entries ---
if 'journal_log' not in st.session_state:
    st.session_state.journal_log = pd.DataFrame(columns=['date', 'entry', 'prediction', 'probability', 'explanation_html'])

# --- Main App Interface ---
st.title("Advanced AI Mental Health Journal")
st.write("This journal uses AI to find patterns and now includes **Explainable AI** to show you which words influence the analysis.")

journal_entry = st.text_area("How was your day?", height=200, key="journal_input")

if st.button("Analyze and Log Entry"):
    if journal_entry:
        # 1. Clean the entry for prediction
        cleaned_entry_for_pred = clean_text(journal_entry)
        
        # 2. Get prediction and probability
        prediction = model.predict([cleaned_entry_for_pred])[0]
        probability = model.predict_proba([cleaned_entry_for_pred])[0][1]

        # --- NEW: Generate the LIME explanation ---
        # We use the original entry for the explanation to preserve context
        explanation = explainer.explain_instance(journal_entry, predictor, num_features=10, labels=(1,))
        explanation_html = explanation.as_html()

        # 3. Store the new entry along with its explanation
        today = pd.to_datetime('today').date()
        new_log = pd.DataFrame({
            'date': [today],
            'entry': [journal_entry],
            'prediction': [prediction],
            'probability': [probability],
            'explanation_html': [explanation_html] # Store the HTML
        })
        st.session_state.journal_log = pd.concat([st.session_state.journal_log, new_log], ignore_index=True)
        st.success("Entry logged successfully!")

    else:
        st.warning("Please write an entry.")

# --- Dashboard for Pattern Visualization ---
st.header("Your Journaling Patterns")

if not st.session_state.journal_log.empty:
    log_df = st.session_state.journal_log.copy()
    log_df['date'] = pd.to_datetime(log_df['date'])

    # Time Series Chart
    st.subheader("Distress Probability Trend")
    st.line_chart(log_df.rename(columns={'date':'index'}).set_index('index')['probability'])
    
    # --- NEW: Display recent entries with their explanations ---
    st.subheader("Recent Entry Analysis")
    # Sort by date to show the latest entry first
    recent_entries = log_df.sort_values(by='date', ascending=False)
    
    if not recent_entries.empty:
        for i, row in recent_entries.head(3).iterrows():
            with st.expander(f"**{row['date'].strftime('%Y-%m-%d')}** | Distress Probability: {row['probability']:.2f}"):
                st.markdown("#### Why the model made this prediction:")
                st.write("The words highlighted in green point towards potential distress, while red points away.")
                # Render the explanation HTML
                st.components.v1.html(row['explanation_html'], height=250, scrolling=True)
                st.markdown("#### Your Full Entry:")
                st.write(row['entry'])
    else:
        st.info("No recent entries have been flagged.")
else:
    st.info("Your dashboard will appear here once you start logging entries.")

