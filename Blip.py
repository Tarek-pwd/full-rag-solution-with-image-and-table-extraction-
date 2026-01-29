from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import cv2
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
def caption_image(img):
    inputs = processor(images=img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption 