import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dropout, Bidirectional, LSTM, Dense, Embedding
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load and Clean Data
df = pd.read_csv('data/Train.csv')
# Remove rows with missing text or labels
df = df.dropna(subset=['safe_text', 'label']) 
# Convert labels to integers and shift from [-1, 0, 1] to [0, 1, 2]
df['label'] = df['label'].astype(int) + 1    

# 2. Text Preprocessing
# Initialize tokenizer with a 10,000 word limit and an Out-Of-Vocabulary token
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['safe_text'].astype(str))

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(df['safe_text'].astype(str))
# Pad sequences to a uniform length of 100
padded = pad_sequences(sequences, maxlen=100, padding='post')

# 3. Enhanced Bidirectional LSTM Model
model = tf.keras.Sequential([
    # Embedding layer to turn integers into dense vectors
    tf.keras.layers.Embedding(10000, 64, input_length=100),
    # First Bidirectional LSTM layer (returns sequences for the next layer)
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.5), # Regularization to prevent overfitting
    # Second Bidirectional LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.5),
    # Dense hidden layer
    tf.keras.layers.Dense(24, activation='relu'),
    # Output layer for 3 classes (Negative, Neutral, Positive)
    tf.keras.layers.Dense(3, activation='softmax')
])

# Configure the optimizer with a stable learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 4. Training with Monitoring (Early Stopping)
# Stop training if val_loss doesn't improve for 3 consecutive epochs
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starting model training...")
model.fit(
    padded, 
    df['label'], 
    epochs=20, 
    validation_split=0.2, 
    batch_size=32,
    callbacks=[early_stop]
)

# 5. Save the Final Model
model.save('models/sentiment_lstm.h5')
print("Model trained and saved successfully in the models/ folder!")