import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from io import BytesIO, StringIO
from PIL import Image
import base64
import numpy as np
import pandas as pd
from gtts import gTTS
import re
import asyncio

# Import your AI assistant functions here
from model.inference import generate_strategy, generate_image, describe_image
from influencer_selector import detect_domain_and_country, get_influencers
from generator import SocialPostGenerator
from transformers import pipeline
import speech_recognition as sr

# Initialize AI tools
generator = SocialPostGenerator("generated_posts.csv")
recognizer = sr.Recognizer()
sentiment_pipeline = pipeline("sentiment-analysis", device=0)

# FastAPI app
app = FastAPI()

# Allow CORS for frontend calls (replace * with your domain in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper: clean text
def clean_message(text):
    cleaned = re.sub(r'[^\w\s]', '', text)
    return cleaned.lower().strip()

# Chat request model
class ChatRequest(BaseModel):
    message: str
    budget: Optional[float] = None

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    message = request.message
    budget = request.budget
    message_clean = clean_message(message)
    
    # Simplified unified_bot logic, adapted for async
    response = None
    
    # Image generation
    if "generate" in message_clean and "image" in message_clean:
        response = handle_image_generation(message)
    elif any(kw in message_clean for kw in ["generate gif", "create animation", "make animated"]):
        response = handle_gif_generation(message)
    elif any(kw in message_clean for kw in ["marketing strategy", "retain customers",
                                            "customer retention", "acquisition plan",
                                            "growth strategy", "develop a plan"]):
        response = handle_marketing_strategy(message)
    elif any(kw in message_clean for kw in ["influencer", "promote", "marketing", "advertise"]):
        response = handle_influencer_recommendations(message, budget)
    elif any(kw in message_clean for kw in ["generate post", "create post", "social post"]):
        response = handle_social_post_generation(message)
    else:
        response = ("🤖 I can help with:\n"
                    "- Marketing strategies\n"
                    "- Sentiment analysis\n"
                    "- AI image generation\n"
                    "- GIF/Animation creation\n"
                    "- Influencer recommendations\n"
                    "- Social media posts (upload an image or describe your post)")
    return {"response": response}

@app.post("/api/sentiment")
async def sentiment_analysis(file: UploadFile = File(...)):
    try:
        content = await file.read()
        try:
            content_str = content.decode('utf-8')
            df = pd.read_csv(StringIO(content_str))
        except UnicodeDecodeError:
            df = pd.read_excel(BytesIO(content))
        required_columns = ['comments', 'comment', 'text', 'review']
        found_column = next((col for col in required_columns if col in df.columns), None)
        if not found_column:
            return JSONResponse(status_code=400, content={"error": f"CSV must contain one of: {', '.join(required_columns)}"})
        results = []
        batch_size = 5
        for i in range(0, len(df), batch_size):
            batch = df[found_column].iloc[i:i + batch_size].tolist()
            analysis = generate_strategy(
                f"Classify these comments as ONLY 'positive' or 'negative':\n" +
                "\n".join([f"{j + 1}. {text}" for j, text in enumerate(batch)])
            )
            ratings = [1 if "positive" in line.lower() else 0 for line in analysis.split("\n") if line.strip()]
            results.extend(ratings[:len(batch)])
        positive_count = sum(results)
        total = len(results)
        positive_pct = (positive_count / total) * 100 if total > 0 else 0
        visual = "✅ " * int(positive_pct // 20) + "❌ " * int((100 - positive_pct) // 20)
        summary = (f"📊 Sentiment Analysis Results:\n"
                   f"-------------------------\n"
                   f"Total Comments Analyzed: {total}\n"
                   f"Positive: {positive_pct:.1f}% {visual}\n"
                   f"Negative: {100 - positive_pct:.1f}%\n\n"
                   f"Majority: {'POSITIVE' if positive_pct >= 50 else 'NEGATIVE'}")
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/analyze-image")
async def analyze_image(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(BytesIO(contents))
    # Example: describe image
    description = describe_image(img)
    return {"description": description}

@app.post("/api/tts")
async def tts_endpoint(text: str = Form(...)):
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join("/tmp", filename)
    try:
        tts = gTTS(text)
        tts.save(audio_path)
        return FileResponse(audio_path, media_type="audio/mpeg", filename=filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Helper functions from your code (non-async wrappers)

def handle_image_generation(message):
    try:
        img = generate_image(message)
        if isinstance(img, Image.Image):
            return image_to_base64_str(img)
        else:
            return f"❌ Image generation returned unexpected type: {type(img)}"
    except Exception as e:
        return f"❌ Image generation error: {e}"

def handle_gif_generation(message):
    return generate_animated_image(message)

def handle_marketing_strategy(message):
    return generate_marketing_strategy(message)

def handle_influencer_recommendations(message, budget):
    domain, country = detect_domain_and_country(message)
    df = get_influencers(domain, country, budget)
    if df.empty:
        return "⚠️ No influencers found."
    return f"🤝 Recommended Influencers:\n\n{df.to_markdown(index=False)}"

def handle_social_post_generation(message, image=None):
    try:
        if image is not None:
            description = describe_image(image)
            if not message.strip():
                message = "Create a social media post for this image"
            result = generator.generate_posts(f"{message}\n\nImage description: {description}")
        else:
            result = generator.generate_posts(message)
        return f"📱 Generated Post:\n\n{result}"
    except Exception as e:
        return f"❌ Error generating social post: {str(e)}"

def image_to_base64_str(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"![Generated Image](data:image/png;base64,{encoded})"
