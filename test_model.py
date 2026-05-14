import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer

# 1. Charger le modèle sauvegardé
model = tf.keras.models.load_model('models/sentiment_lstm.h5')

# 2. Recréer le tokenizer (doit être identique à l'entraînement)
df = pd.read_csv('data/Train.csv').dropna(subset=['safe_text'])
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['safe_text'].astype(str))

# 3. Fonction de test
def predict_sentiment(text):
    # Prétraitement de la phrase
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100, padding='post')
    
    # Prédiction
    prediction = model.predict(padded)
    
    # Rappel : nos labels sont décalés de +1 (0: Négatif, 1: Neutre, 2: Positif)
    classes = ['Négatif', 'Neutre', 'Positif']
    import numpy as np
    result = classes[np.argmax(prediction)]
    return result

# 4. Test manuel
phrase = "The vaccine is safe and effective for everyone!"
print(f"Phrase : {phrase}")
print(f"Sentiment prédit : {predict_sentiment(phrase)}")