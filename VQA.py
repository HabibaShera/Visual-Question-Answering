from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image


def getAnswer(question, image_path):
    """
    This function return result of the entered data

    -Parameters:
    question : str
    image_path : str

    -Returns:
    result of the entered question (str)
    """
    image = Image.open(image_path)

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])

getAnswer('What is the color of the car', 'car.jpg')