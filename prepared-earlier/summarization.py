# Text summarization models
# Complete as part of workshop, find a model or multiple for comparisons
# e.g. go here https://huggingface.co/facebook/bart-large-cnn
import gradio as gr
import textwrap
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration

bart_model = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


# https://blog.paperspace.com/bart-model-for-text-summarization-part1/


def summarize(text):
    return bart_model(text, max_length=150, min_length=30)


# https://www.projectpro.io/article/transformers-bart-model-explained/553
def summarize_using_tokenizer(text):
    inputs = tokenizer.batch_encode_plus([text], return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, min_length=30, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == '__main__':
    demo = gr.Interface(fn=summarize, inputs="text", outputs="text")
    demo.launch()
