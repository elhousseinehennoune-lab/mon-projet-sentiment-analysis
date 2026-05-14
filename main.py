import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

# 1. Load your locally trained LSTM model
model_path = 'models/sentiment_lstm.h5'
model = tf.keras.models.load_model(model_path)

# 2. Recreate the Tokenizer (must be identical to EDA.py)
# Using Train.csv to ensure the word dictionary is the same
df_train = pd.read_csv('data/Train.csv').dropna(subset=['safe_text'])
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df_train['safe_text'].astype(str))

def preprocess(text):
    # Logic for cleaning usernames and links
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def sentiment_analysis(text):
    text = preprocess(text)
    
    # Text transformation for LSTM (Tokenization + Padding)
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    
    # Prediction with your model
    prediction = model.predict(padded)[0]
    
    # English Labels: 0: Negative, 1: Neutral, 2: Positive
    labels = ['Negative', 'Neutral', 'Positive']
    
    # Create dictionary for Gradio output
    return {labels[i]: float(prediction[i]) for i in range(len(labels))}

# 3. Gradio Interface (English Version)
demo = gr.Interface(
    theme=gr.themes.Base(),
    fn=sentiment_analysis,
    inputs=gr.Textbox(
        label="Tweet Input", 
        placeholder="Type your tweet here..."
    ),
    outputs=gr.Label(label="Sentiment Prediction"),
    title='COVID-19 Vaccine Sentiment Analysis (LSTM)',
    description='This application uses a custom Bidirectional LSTM model trained with over 90% accuracy to analyze vaccination-related sentiments.',
    examples=[
        ["The COVID Vaccine saves lives!"],
        ["The vaccine is terrible. It can lead to early death"],
        ["I'm not sure. Maybe i'll get my booster vaccine shot"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True)