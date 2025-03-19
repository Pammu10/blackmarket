import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/full_dataset/goemotions_1.csv")

# Define emotion labels
emotion_labels = ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity',
                  'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
                  'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                  'remorse', 'sadness', 'surprise', 'neutral']

# Map emotion labels to their respective indices
df['emotions'] = df[emotion_labels].apply(lambda x: [emotion_labels[i] for i in range(len(x)) if x[i] == 1], axis=1)

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['text'] = df['comment'].apply(clean_text)

# Convert labels to one-hot encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['emotions'])

# Tokenization
tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, maxlen=50)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save tokenizer for later use
import pickle
with open("tokenizer.pkl", "wb") as handle:
    pickle.dump(tokenizer, handle)
