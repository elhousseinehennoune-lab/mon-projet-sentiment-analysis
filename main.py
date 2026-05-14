from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from scipy.special import softmax
import gradio as gr


model_path = f"Azie88/COVID_Vaccine_Tweet_sentiment_analysis_roberta"

tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def sentiment_analysis(text):
    text = preprocess(text)

    # PyTorch-based models
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores_ = output[0][0].detach().numpy()
    scores_ = softmax(scores_)

    # Format output dict of scores
    labels = ['Negative', 'Neutral', 'Positive']
    scores = {l:float(s) for (l,s) in zip(labels, scores_) }

    return scores


demo = gr.Interface(theme=gr.themes.Base(),
    fn=sentiment_analysis,
    inputs=gr.Textbox(placeholder="Write your tweet here..."),
    outputs="label",
    # interpretation="default",
    examples=[["The COVID Vaccine saves lives!"],
              ["The Vaccination is not necessary for young people"],
              ["The vaccine is terrible. It can lead to early death"],
              ["I'm not sure. Maybe i'll get my booster vaccine shot"]],
    title='COVID Vaccine Sentiment Analysis app',
    description='This app assesses if a tweet related to vaccinations has a positive, neutral or negative sentiment.'
              )

demo.launch(share=True)