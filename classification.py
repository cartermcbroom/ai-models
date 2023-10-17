import gradio as gr
from transformers import pipeline

classifier = pipeline(model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)


def classify(prompt):
    return classifier(prompt)[0]


if __name__ == '__main__':
    demo = gr.Interface(fn=classify,
                        inputs="text",
                        outputs="text",
                        title="Classifying emotion in text using distilbert-base-uncased-emotion",
                        description="Will score some text on different emotions, e.g. joy, happy, sad",
                        examples="I love gammon")
    demo.launch()
