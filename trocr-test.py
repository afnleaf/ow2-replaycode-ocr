from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# load image from the IAM dataset
#url = "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
#image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

images = []

for i in range(7):
    print(i)
    images.append(Image.open(f"./output/after_{i}.png").convert("RGB"))
#= Image.open("./output/after_0.png").convert("RGB")

for image in images:
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text.upper())

