import pandas as pd
import matplotlib.pyplot as plt

# Charger les données depuis votre dossier data (image_79a333.png)
df = pd.read_csv('data/Train.csv')

# Regarder la distribution des sentiments
print(df['label'].value_counts())

# Calculer la longueur des tweets pour le LSTM
df['length'] = df['safe_text'].str.split().str.len()
print(f"Longueur moyenne : {df['length'].mean()}")