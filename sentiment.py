# Sentiment analysis models
# Task - Run demo and update the function to test the other models.
# Try different models and different inputs.
from transformers import pipeline
import gradio as gr

siebert_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
fine_tuned_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tweet_specific_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")


def sentiment(text):
    return siebert_model(text)


if __name__ == '__main__':
    demo = gr.Interface(
        fn=sentiment,
        inputs="text",
        outputs="text",
        title="Sentiment analysis using BERT models",
        description="Will determine if a texts emotional tone is positive, negative or neutral",
        examples="I love fishing"
    )
    demo.launch()
