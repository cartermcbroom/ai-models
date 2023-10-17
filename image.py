import gradio as gr
from diffusers import DiffusionPipeline
from transformers import pipeline

image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
text_to_image_pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")


def caption_image(image_url):
    return image_captioning_pipeline(image_url)[0]['generated_text']


if __name__ == '__main__':
    gr.close_all()
    demo = gr.Interface(fn=caption_image,
                        inputs=[gr.Textbox(label="Image url")],
                        outputs=[gr.Textbox(label="Caption")],
                        title="Image Captioning with BLIP",
                        description="Caption any image using the BLIP model",
                        allow_flagging="never",
                        examples=[
                            "https://media.istockphoto.com/id/483293702/photo/laughing-donkey.webp?s=2048x2048&w=is&k=20&c=0kIpTrRV9rUufHE6jR_PqhWqqQBC4r1A-yyjVda5TUU="])
    demo.launch()
