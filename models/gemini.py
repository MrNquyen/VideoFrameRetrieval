import os
import torch
import numpy as np
import google.generativeai as genai
from torchvision import transforms
from PIL import Image
from dotenv import load_dotenv
from utils.template import PARSE_INSTRUCTION_TEMPLATE, PARSE_QUESTION_TEMPLATE, PARSE_OUTPUT_TEMPLATE
from utils.template import PARAPHRASE_INSTRUCTION_TEMPLATE, PARAPHRASE_QUESTION_TEMPLATE, PARAPHRASE_OUTPUT_TEMPLATE
from utils.template import CAPTION_INSTRUCTION_TEMPLATE, CAPTION_QUESTION_TEMPLATE, CAPTION_OUTPUT_TEMPLATE
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

class Gemini:
    def __init__(self):
        self.model = model

    def convert_image_type(self, image):
        if type(image) == np.ndarray:
            image = Image.fromarray(image)
        elif type(image) == torch.Tensor:
            to_pil = transforms.ToPILImage()
            image = to_pil(image)
        return image

    #-- CAPTIONING
    def captioning(self, images):
        images = [self.convert_image_type(image) for image in images]
        caption_response = self.model.generate_content([
            CAPTION_INSTRUCTION_TEMPLATE,
            CAPTION_QUESTION_TEMPLATE,
            *images,
            CAPTION_OUTPUT_TEMPLATE
        ])
        image_captions = caption_response.text.split("\n")
        while "" in image_captions:
            image_captions.remove("")
        return image_captions


    #-- PARAPHRASING
    def parapharasing(self, sentences):
        paraphrase_response = self.model.generate_content([
            PARAPHRASE_INSTRUCTION_TEMPLATE,
            PARAPHRASE_QUESTION_TEMPLATE,
            *sentences,
            PARAPHRASE_OUTPUT_TEMPLATE
        ])
        paraphrase_captions = paraphrase_response.text.split("\n")
        while "" in paraphrase_captions:
            paraphrase_captions.remove("")
        return paraphrase_captions


    #-- PARSING
    def parsing(self, captions):
        captions = ["Caption:" + caption for caption in captions]
        parse_response = self.model.generate_content([
            PARSE_INSTRUCTION_TEMPLATE,
            PARSE_QUESTION_TEMPLATE,
            *captions,
            PARSE_OUTPUT_TEMPLATE
        ])
        parse_captions = parse_response.text.split("\n")
        while "" in parse_captions:
            parse_captions.remove("")
        return parse_captions