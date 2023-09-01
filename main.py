from transformers import pipeline

image_captioning_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")


def caption_image(image_url):
    output = image_captioning_pipeline(image_url)
    print(output[0]['generated_text'])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = "https://free-images.com/lg/ee7b/cow_animal_cow_head.jpg"
    caption_image(image)

# You can use this model for conditional and un-conditional image captioning ?

# https://huggingface.co/Salesforce/blip-image-captioning-base


