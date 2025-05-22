import os
import uuid
from PIL import Image
import base64
from io import BytesIO, StringIO
import re
import pandas as pd
import speech_recognition as sr
import gradio as gr
from huggingface_hub import InferenceClient
from spaces import GPU
import numpy as np
import imageio
from typing import List
from gtts import gTTS

from model.inference import generate_strategy, generate_image, describe_image
from influencer_selector import detect_domain_and_country, get_influencers
from generator import SocialPostGenerator

HF_TOKEN = os.getenv("HF_TOKEN")
client = InferenceClient(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    token=HF_TOKEN if HF_TOKEN else None
)
generator = SocialPostGenerator("generated_posts.csv")
recognizer = sr.Recognizer()

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", device=0)


def generate_post_from_sentiment(text: str) -> str:
    sentiment_result = sentiment_pipeline(text)[0]
    sentiment = sentiment_result['label'].lower()

    if sentiment == "positive":
        base_prompt = "Create an excited social media post about fashion trends."
    elif sentiment == "negative":
        base_prompt = "Create a concerned post highlighting a problem."
    else:
        base_prompt = "Create an informative post."

    generated_post = generator.generate_posts(f"{base_prompt}\n\nSentiment: {sentiment}\nText: {text}")
    return f"🧠 Detected Sentiment: *{sentiment.capitalize()}*\n\n📱 Generated Post:\n\n{generated_post}"


def text_to_speech(text, filename=None):
    if not filename:
        filename = f"tts_{uuid.uuid4().hex}.mp3"
    audio_path = os.path.join("/tmp", filename)
    try:
        tts = gTTS(text)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"TTS generation error: {e}")
        return None


def generate_gif(frames: List[Image.Image], output_path: str = "output.gif", duration: int = 200) -> str:
    try:
        os.makedirs("/tmp/gifs", exist_ok=True)
        output_path = f"/tmp/gifs/{uuid.uuid4().hex}.gif"
        frame_arrays = [np.array(frame.convert('RGB')) for frame in frames]
        imageio.mimsave(
            output_path,
            frame_arrays,
            format='GIF',
            duration=duration / 1000,
            loop=0
        )
        return output_path
    except Exception as e:
        print(f"Error generating GIF: {e}")
        return None


def generate_animated_image(prompt: str, num_frames: int = 5) -> str:
    try:
        frames = []
        for i in range(num_frames):
            frame_prompt = f"{prompt} - frame {i + 1}/{num_frames}"
            img = generate_image(frame_prompt)
            if isinstance(img, Image.Image):
                frames.append(img)
        if len(frames) < 2:
            return "❌ Need at least 2 frames to create a GIF"
        gif_path = generate_gif(frames)
        if gif_path and os.path.exists(gif_path):
            with open(gif_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode()
            return f"![Generated GIF](data:image/gif;base64,{encoded})"
        return "❌ Failed to generate GIF"
    except Exception as e:
        return f"❌ GIF generation error: {e}"


def transcribe_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        print(f"Audio path invalid or missing: {audio_path}")
        return "❌ Invalid audio file."

    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            transcript = recognizer.recognize_google(audio)
            return f"{transcript} (voice_input)"
    except sr.UnknownValueError:
        return "❌ Could not understand audio."
    except sr.RequestError as e:
        return f"❌ Speech recognition service error: {e}"
    except Exception as e:
        return f"❌ General speech recognition error: {e}"


def image_to_base64_str(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"![Generated Image](data:image/png;base64,{encoded})"


def generate_marketing_strategy(prompt: str) -> str:
    system_prompt = """[INST] <<SYS>>
You are a Chief Marketing Officer. Always respond with this EXACT structure:

Answer:
*[Strategy Title]*

1. CURRENT LANDSCAPE:
   - [Bullet 1]
   - [Bullet 2]

2. RECOMMENDED STRATEGY:
   Step 1: [Action] → [Rationale]
   Step 2: [Action] → [Rationale]

3. EXECUTION ROADMAP:
   - Week 1-2: [Tasks]
   - Week 3-4: [Tasks]
<</SYS>>[/INST]"""
    try:
        response = client.text_generation(
            f"{system_prompt}\n\nUser Request: {prompt}",
            max_new_tokens=1200,
            temperature=0.7
        )
        return response
    except Exception as e:
        return f"❌ Failed to generate strategy (API Error): {str(e)}"


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


def handle_sentiment_analysis(message, file):
    try:
        if hasattr(file, 'read'):
            content = file.read()
        else:
            with open(file, 'rb') as f:
                content = f.read()
        try:
            content_str = content.decode('utf-8')
            df = pd.read_csv(StringIO(content_str))
        except UnicodeDecodeError:
            df = pd.read_excel(BytesIO(content))
        required_columns = ['comments', 'comment', 'text', 'review']
        found_column = next((col for col in required_columns if col in df.columns), None)
        if not found_column:
            return f"❌ CSV must contain one of: {', '.join(required_columns)}"
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
        return summary
    except Exception as e:
        return f"❌ Error analyzing file: {str(e)}"


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


def clean_message(text):
    cleaned = re.sub(r'[^\w\s]', '', text)
    return cleaned.lower().strip()


def get_help_message():
    return (
        "🤖 I can help with the following tasks:\n"
        "- Generate images or GIFs based on prompts\n"
        "- Analyze sentiment from text or uploaded files\n"
        "- Create marketing strategies\n"
        "- Recommend influencers based on your domain and budget\n"
        "- Generate social media posts\n"
        "- Transcribe voice messages\n\n"
        "Just describe what you'd like to do!"
    )


@GPU
def unified_bot(message, history, file=None, image=None):
    history = history or []
    message_clean = clean_message(message)

    budget = None
    budget_match = re.search(r"\$?(\d{2,5})", message)
    if budget_match:
        try:
            budget = float(budget_match.group(1))
        except ValueError:
            pass

    image_keywords = ["image", "picture", "photo", "illustration", "visual", "artwork", "drawing"]
    gif_keywords = ["gif", "animation", "animated", "moving image", "cinemagraph"]
    sentiment_keywords = ["sentiment", "analysis", "feedback", "reviews", "comments", "survey"]
    strategy_keywords = ["strategy", "plan", "roadmap", "approach", "campaign", "tactic"]
    influencer_keywords = ["influencer", "creator", "promoter", "ambassador", "blogger", "youtuber"]
    post_keywords = ["post", "content", "tweet", "caption", "social media", "instagram", "facebook", "linkedin"]
    audio_keywords = ["voice", "transcribe", "audio", "speech"]

    try:
        if (any(kw in message_clean for kw in image_keywords) and not any(kw in message_clean for kw in gif_keywords)):
            if "generate" in message_clean or "create" in message_clean or "make" in message_clean:
                response = handle_image_generation(message)
            else:
                response = "Would you like me to generate an image? Please include 'create' or 'generate' in your request."

        elif any(kw in message_clean for kw in gif_keywords):
            response = handle_gif_generation(message)

        elif any(kw in message_clean for kw in sentiment_keywords) and file is not None:
            if hasattr(file, 'name') and file.name.endswith(('.csv', '.xlsx', '.json')):
                response = handle_sentiment_analysis(message, file)
            else:
                response = "❌ Please upload a CSV, Excel, or JSON file for analysis"

        elif any(kw in message_clean for kw in strategy_keywords):
            response = handle_marketing_strategy(message)

        elif any(kw in message_clean for kw in influencer_keywords):
            response = handle_influencer_recommendations(message, budget)

        elif any(kw in message_clean for kw in post_keywords):
            response = handle_social_post_generation(message, image)

        elif any(kw in message_clean for kw in audio_keywords) and file:
            audio_path = file.name if hasattr(file, "name") else file
            transcription = transcribe_audio(audio_path)
            response = generate_post_from_sentiment(transcription)

        else:
            response = get_help_message()

    except Exception as e:
        response = f"""⚠️ Oops! I encountered an error processing your request.

Please try:
- Rephrasing your question
- Checking your file format
- Being more specific

Error details: {str(e)}"""

    # Audio generation with improved handling
    audio_path = None
    if isinstance(response, str) and not response.startswith(("![Generated", "❌", "⚠️")):
        audio_keywords = ["speak", "say", "read", "audio", "voice", "hear", "listen", "vocal"]
        if any(kw in message_clean for kw in audio_keywords):
            try:
                # Create temp directory if it doesn't exist
                os.makedirs("/tmp/audio", exist_ok=True)
                
                # Generate unique filename
                audio_filename = f"response_{uuid.uuid4().hex}.mp3"
                audio_path = os.path.join("/tmp/audio", audio_filename)
                
                # Generate and save audio
                tts = gTTS(text=response, lang='en')
                tts.save(audio_path)
                
                # Verify file was created
                if not os.path.exists(audio_path):
                    raise FileNotFoundError("Audio file was not created")
                    
            except Exception as e:
                print(f"Audio generation error: {str(e)}")
                response += f"\n\n⚠️ Couldn't generate audio: {str(e)}"
                audio_path = None

    history.append([message, response])
    return history, audio_path


def get_help_message():
    return """🤖 **How I Can Help You:**

🎨 **Image Creation**  
   Try: "Generate an image of..."  
   "Create a product photo of..."  
   "Make an illustration showing..."  

🔄 **Animations**  
   Try: "Create a GIF of..."  
   "Make an animated version of..."  

📊 **Data Analysis**  
   Try: "Analyze these survey results" (upload file)  
   "What's the sentiment in these reviews?"  

📈 **Marketing Strategy**  
   Try: "Develop a marketing plan for..."  
   "Customer retention strategy for..."  
   "30-day growth roadmap"  

📢 **Influencer Marketing**  
   Try: "Find fashion influencers in France"  
   "Recommend tech creators with 10K-50K followers"  

📱 **Social Media Content**  
   Try: "Write a Twitter thread about..."  
   "Create Instagram captions for this product" (upload image)  

💡 **Tip**: Add details for better results!  
   Instead of: "Make a post"  
   Try: "Create a LinkedIn post announcing our new AI features, focusing on benefits for small businesses"  
"""

@GPU
def unified_bot_with_audio(text_or_audio, history=None, file=None, image=None):
    history = history or []
    is_audio = isinstance(text_or_audio, str) and text_or_audio.endswith((".wav", ".mp3"))
    transcript = transcribe_audio(text_or_audio) if is_audio else text_or_audio

    if not transcript or transcript.startswith("❌"):
        history.append(["(voice input)", transcript or "❌ Failed to transcribe audio."])
        return history, None

    transcript_lower = transcript.lower()

    if any(kw in transcript_lower for kw in ["generate post", "create post", "social post"]):
        try:
            post_output = generator.generate_posts(transcript)
            audio_summary = text_to_speech(post_output)
            history.append([f"{transcript}", f"{post_output}"])
            return history, audio_summary
        except Exception as e:
            history.append([f"{transcript}", f"❌ Error generating post: {str(e)}"])
            return history, None

    default = ("🤖 I can help with:\n"
               "- Marketing strategies\n"
               "- Sentiment analysis\n"
               "- AI image generation\n"
               "- GIF/Animation creation\n"
               "- Influencer recommendations\n"
               "- Social media posts (upload an image or describe your post)")
    history.append([f"{transcript}", default])
    audio_path = text_to_speech(default)
    return history, audio_path


# Gradio UI

# Replace your current Gradio UI section with this:

# Custom CSS for the theme
custom_css = """
:root {
    --primary: #9370DB !important;
    --primary-dark: #7B68EE !important;
    --secondary: #E6E6FA !important;
}

body {
    background: white !important;
    font-family: 'Poppins', sans-serif !important;
}

.gradio-container {
    background: white !important;
    border-radius: 16px !important;
}

.gr-button-primary {
    background: var(--primary) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
}

.gr-button-primary:hover {
    background: var(--primary-dark) !important;
}

.gr-button-secondary {
    background: var(--secondary) !important;
    color: var(--primary) !important;
    border-radius: 8px !important;
}

.gr-interface {
    padding: 24px !important;
}

.gr-chatbot {
    border-radius: 16px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
    border: 1px solid #E6E6FA !important;
}

.gr-input, .gr-textbox, .gr-file, .gr-image {
    border-radius: 12px !important;
    border: 1px solid #E6E6FA !important;
}

.gr-tabs {
    border-radius: 16px !important;
}

h1, h2, h3, h4 {
    color: var(--primary) !important;
    font-weight: 600 !important;
}
"""

# Gradio UI with Audio Fixes
with gr.Blocks(
    title="MarketMind AI Assistant",
    css=custom_css,
    theme=gr.themes.Default(
        primary_hue="purple",
        secondary_hue="purple",
        neutral_hue="gray"
    )
) as demo:
    # Header Section
    with gr.Row():
        gr.Markdown("""
        <div style="text-align: center; width: 100%">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem">✨ MarketMind AI</h1>
            <p style="color: #9370DB; font-size: 1.1rem">Your Intelligent Marketing Assistant</p>
        </div>
        """)
    
    # Main Content
    with gr.Row():
        # Left Sidebar (Inputs)
        with gr.Column(scale=1, min_width=300):
            with gr.Group():
                gr.Markdown("### 📋 Input Options")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎤 Voice Input",
                    elem_classes="purple-border"
                )
                upload_btn = gr.File(
                    label="📊 Upload Data",
                    file_types=[".csv", ".xlsx"],
                    elem_classes="purple-border"
                )
                image_input = gr.Image(
                    type="filepath",
                    label="🖼️ Upload Image",
                    elem_classes="purple-border"
                )
            
            with gr.Accordion("💡 Quick Tips", open=False):
                gr.Markdown("""
                - **Ask for marketing strategies**
                - **Upload CSVs for sentiment analysis**
                - **Say 'generate image' for AI art**
                - **Try 'create social media post'**
                - **Say 'speak response' for audio output**
                """)

        # Right Panel (Chat)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 MarketMind Chat",
                height=500,
                bubble_full_width=False,
                avatar_images=(
                    "https://i.imgur.com/7WqjJz3.png",  # User avatar
                    "superhero.png"   # Bot avatar
                )
            )
            
            with gr.Row():
                text_input = gr.Textbox(
                    placeholder="Ask MarketMind anything about marketing...",
                    label="Type your message",
                    container=False,
                    scale=7
                )
                submit_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                    min_width=100
                )
            
            # Enhanced Audio Player
            with gr.Accordion("🔊 Voice Response", open=True):
                voice_preview = gr.Audio(
                    label="Listen to response",
                    type="filepath",
                    visible=True,
                    interactive=False,
                    elem_classes="audio-player"
                )
                auto_play_toggle = gr.Checkbox(
                    label="Auto-play audio responses",
                    value=True,
                    interactive=True
                )

    # Interaction Logic with Audio Fixes
    def process_input(message, history, file, image, auto_play):
        history = history or []
        response_history, audio_path = unified_bot(message, history, file, image)
        
        # Auto-play logic
        if auto_play and audio_path:
            return response_history, audio_path, gr.update(visible=True, value=audio_path)
        elif audio_path:
            return response_history, audio_path, gr.update(visible=True, value=audio_path)
        else:
            return response_history, None, gr.update(visible=False)

    # Connect all components
    submit_btn.click(
        fn=process_input,
        inputs=[text_input, chatbot, upload_btn, image_input, auto_play_toggle],
        outputs=[chatbot, voice_preview, voice_preview]
    ).then(
        fn=lambda: ("", None, None),
        outputs=[text_input, audio_input, image_input]
    )

    text_input.submit(
        fn=process_input,
        inputs=[text_input, chatbot, upload_btn, image_input, auto_play_toggle],
        outputs=[chatbot, voice_preview, voice_preview]
    ).then(
        fn=lambda: ("", None, None),
        outputs=[text_input, audio_input, image_input]
    )

    audio_input.change(
        fn=process_input,
        inputs=[audio_input, chatbot, upload_btn, image_input, auto_play_toggle],
        outputs=[chatbot, voice_preview, voice_preview]
    )

    upload_btn.change(
        fn=lambda f: f"📁 Ready to analyze: {f.name}",
        inputs=upload_btn,
        outputs=text_input
    )

    image_input.change(
        fn=lambda img: "📷 generate post for this image",
        inputs=image_input,
        outputs=text_input
    )

if __name__ == "__main__":
    demo.launch(
        favicon_path="https://i.imgur.com/4XvQ2bQ.png",
        allowed_paths=["."]
    )