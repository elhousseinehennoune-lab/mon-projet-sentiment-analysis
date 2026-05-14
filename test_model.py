import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np

# 1. Load the saved model
model_path = 'models/sentiment_lstm.h5'
model = tf.keras.models.load_model(model_path)

# 2. Recreate the tokenizer (must be identical to the training setup)
# We load the training data to ensure the vocabulary mapping matches
df = pd.read_csv('data/Train.csv').dropna(subset=['safe_text'])
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['safe_text'].astype(str))

# 3. Prediction Function
def predict_sentiment(text):
    # Preprocessing the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    
    # Run prediction
    prediction = model.predict(padded)
    
    # Recall: Labels are shifted by +1 (0: Negative, 1: Neutral, 2: Positive)
    classes = ['Negative', 'Neutral', 'Positive']
    
    # Get the class with the highest probability
    result = classes[np.argmax(prediction)]
    return result

# 4. Manual Test
if __name__ == "__main__":
    test_phrase = "The vaccine is safe and effective for everyone!"
    print(f"Input Phrase: {test_phrase}")
    print(f"Predicted Sentiment: {predict_sentiment(test_phrase)}")

    # Try a negative example as well
    negative_phrase = "I am concerned about the unknown long-term risks."
    print(f"\nInput Phrase: {negative_phrase}")
    print(f"Predicted Sentiment: {predict_sentiment(negative_phrase)}")