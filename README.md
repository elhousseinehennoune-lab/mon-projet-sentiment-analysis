🛡️ COVID-19 Vaccine Sentiment Analysis using LSTM

This project implements a Deep Learning solution to classify the sentiment of tweets regarding COVID-19 vaccinations. Using a Bidirectional LSTM (Long Short-Term Memory) network, the model categorizes tweets into three sentiments: Negative, Neutral, or Positive.

🚀 Model Performance
The model was trained on a labeled dataset of vaccination-related tweets. To combat overfitting, techniques such as Dropout and Early Stopping were implemented.

Metric                    Value

Training Accuracy	      91.12%

Validation Accuracy	      68.50%

Architecture         	Bidirectional LSTM

Optimizer	        Adam (Learning Rate: 0.0005)


🛠️ Features
Data Cleaning: Automated removal of missing values (NaN) and preprocessing of special characters.

Text Preprocessing: Custom tokenization and padding (max length: 100) to handle social media language.

Interactive UI: A web interface powered by Gradio for real-time sentiment prediction.

Robustness: Handled label shifting (from [-1, 0, 1] to [0, 1, 2]) for optimal cross-entropy loss calculation.

📂 Project Structure
EDA.py: The training pipeline (Data cleaning, Tokenization, Model architecture, and Training).

main.py: The application script that loads the trained model and launches the Gradio interface.

models/: Directory containing the saved model (sentiment_lstm.h5).

requirements.txt: List of necessary Python libraries.

⚙️ Installation & Usage

1. Clone the repository:
git clone https://github.com/elhousseinehennoune-lab/mon-projet-sentiment-analysis.git
cd mon-projet-sentiment-analysis

2. Install dependencies:
pip install -r requirements.txt

3. Train the model (Optional, as the .h5 file is generated here):
python EDA.py

4. Launch the Web App:
python main.py

.

🧠 Technical Insight
The gap between training and validation accuracy is a common challenge in Twitter sentiment analysis due to the informal nature of the text (slang, sarcasm, and typos). This model uses Bidirectional layers to understand the context of a word based on both its preceding and following words, which is crucial for capturing the nuances of vaccination debates.
