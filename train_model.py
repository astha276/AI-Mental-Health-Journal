
import pandas as pd
import numpy as np
import re
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import BertModel, BertTokenizer

print("Step 1: Loading and Preprocessing Data...")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- UPDATED SECTION ---
# Load the new dataset and use the correct column names
df = pd.read_csv('Mental-Health-Sentiment-Dataset.csv')
df.dropna(subset=['statement'], inplace=True)
df['cleaned_text'] = df['statement'].apply(clean_text)

# Define features (X) and target (y)
X = df['cleaned_text']
y = df['status'] # The label column is named 'labels'
# --- END UPDATED SECTION ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data loaded. Training samples:", len(X_train))

print("\nStep 2: Setting up BERT and TF-IDF components...")

class BertEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("Generating BERT embeddings...")
        embeddings = []
        for text in X:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
            embeddings.append(cls_embedding)
        return np.vstack(embeddings)

tfidf_pipe = Pipeline([('tfidf', TfidfVectorizer(max_features=1000, stop_words='english'))])
bert_pipe = Pipeline([('bert', BertEmbeddings())])
feature_union = FeatureUnion([('tfidf_features', tfidf_pipe), ('bert_features', bert_pipe)])

full_pipeline = Pipeline([
    ('features', feature_union),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

print("\nStep 3: Training the full model... THIS WILL TAKE A LONG TIME.")
full_pipeline.fit(X_train, y_train)

print("\nStep 4: Evaluating the model...")
y_pred = full_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

print("\nStep 5: Saving the trained model...")
joblib.dump(full_pipeline, 'mental_health_model.pkl')
print("\nSUCCESS: Model saved as mental_health_model.pkl")
