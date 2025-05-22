from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch



# Initialize sentiment pipeline once on GPU device 0
sentiment_pipeline = pipeline("sentiment-analysis", device=0)

# Initialize Stable Diffusion pipeline once on GPU with float16 precision
image_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

def generate_strategy(text):
    # Use the globally initialized pipeline
    result = sentiment_pipeline(text)[0]
    return result['label']

def generate_image(prompt):
    # Use the globally initialized Stable Diffusion pipeline
    image = image_pipe(prompt).images[0]
    return image


import requests

from huggingface_hub import InferenceClient
import os

# Initialize client (FREE without API key for some models)
client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1")

from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

# Initialize model and processor globally to avoid reloading
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None

def describe_image(image_path: str) -> str:
    """Generate a detailed description of an image using BLIP-2 model"""
    global processor, model
    
    # Lazy load model to save memory
    if processor is None or model is None:
        print("Loading BLIP-2 model...")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model.to(device)
        print("BLIP-2 model loaded")
    
    try:
        # Load and preprocess image
        raw_image = Image.open(image_path).convert('RGB')
        
        # Generate description
        inputs = processor(raw_image, return_tensors="pt").to(device, torch.float16)
        
        # Generate detailed caption
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True
        )
        
        description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Generate additional details with a different prompt
        prompt = "Question: What are the key elements, colors, and style of this image? Answer:"
        inputs = processor(raw_image, text=prompt, return_tensors="pt").to(device, torch.float16)
        
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            early_stopping=True
        )
        
        details = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return f"{description}. {details}"
    
    except Exception as e:
        print(f"Error describing image: {e}")
        return "An interesting image that would make a great social media post"
    




