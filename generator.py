import pandas as pd
import re
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

class SocialPostGenerator:
    def __init__(self, dataset_path):
        # Load CSV once
        self.df = pd.read_csv(dataset_path)
        # Extract keywords once and save in df
        self.df['keywords'] = self.df['caption'].apply(self.extract_keywords)
        self.available_topics = self.df['topic'].str.lower().unique()

        # Load models once (use device=0 for GPU if available)
        self.rewriter = pipeline("text2text-generation", model="google/flan-t5-base")  # Add device=0 if GPU
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.enricher = pipeline("text2text-generation", model="google/flan-t5-large")  # Add device=0 if GPU

        # Precompute topic embeddings once
        self.topic_embeddings = {
            topic: self.embedding_model.encode(
                " ".join(self.df[self.df['topic'].str.lower() == topic]['caption'].tolist()),
                convert_to_tensor=True
            )
            for topic in self.available_topics
        }

    def extract_keywords(self, text):
        return set(re.findall(r'\b[a-zA-Z]{4,}\b', text.lower()))

    def reformulate_prompt(self, prompt):
        try:
            reformulated = self.rewriter(
                f"Paraphrase this sentence: {prompt}",
                max_length=60,
                do_sample=False
            )[0]['generated_text']
            print(f"🔁 Reformulated: {reformulated}")
            return reformulated
        except:
            return prompt

    def detect_topic_semantic(self, prompt):
        prompt_emb = self.embedding_model.encode(prompt, convert_to_tensor=True)
        scores = {
            topic: util.cos_sim(prompt_emb, topic_emb).item()
            for topic, topic_emb in self.topic_embeddings.items()
        }
        print("📊 Semantic topic scores:", scores)
        best_topic = max(scores, key=scores.get)
        return best_topic if scores[best_topic] > 0.2 else None

    def detect_topic_keywords(self, prompt, min_score_threshold=1):
        prompt_keywords = self.extract_keywords(prompt)
        topic_scores = {}

        for topic in self.available_topics:
            topic_posts = self.df[self.df['topic'].str.lower() == topic]
            topic_keywords = set().union(*topic_posts['keywords'])
            score = len(prompt_keywords & topic_keywords)
            topic_scores[topic] = score

        print("🔤 Keyword topic scores:", topic_scores)
        best_match = max(topic_scores, key=topic_scores.get)
        if topic_scores[best_match] >= min_score_threshold:
            return best_match

        for topic in self.available_topics:
            if topic in prompt.lower():
                print(f"⚠️ Fallback keyword match: {topic}")
                return topic

        return None

    def enrich_post(self, post_data, num_variants=2):
        enriched_versions = []

        prompt_template = (
            "Write a social media post with a {tone} tone. "
            "Use this caption: '{caption}', include these hashtags: {hashtags}, and end with this call to action: '{cta}'. "
            "Make it engaging, natural, and include some fitting emojis."
        )

        for _ in range(num_variants):
            prompt = prompt_template.format(
                tone=post_data['tone'],
                caption=post_data['caption'],
                hashtags=post_data['hashtags'],
                cta=post_data['cta']
            )

            try:
                enriched = self.enricher(prompt, max_length=80, num_return_sequences=1, do_sample=True)[0]['generated_text']
            except:
                enriched = "⚠️ Could not generate enriched post."

            enriched_versions.append(enriched.strip())

        return enriched_versions

    def generate_posts(self, prompt, num_options=2):
        original_prompt = prompt
        prompt = self.reformulate_prompt(prompt)

        topic = self.detect_topic_semantic(prompt)
        if not topic:
            topic = self.detect_topic_keywords(prompt)

        if not topic:
            return "❌ Aucun sujet pertinent détecté pour ce prompt."

        print(f"✅ Topic détecté: {topic}")
        prompt_keywords = self.extract_keywords(prompt)
        self.df['score'] = self.df['keywords'].apply(lambda x: len(x & prompt_keywords))

        relevant_posts = self.df[self.df['topic'].str.lower() == topic]
        relevant_posts = relevant_posts.sort_values('score', ascending=False)
        relevant_posts = relevant_posts.drop_duplicates(subset='caption')
        selected_posts = relevant_posts.head(num_options).to_dict('records')

        output = []
        for i, post in enumerate(selected_posts, 1):
            output.append(
                f"\nOption {i}\n"
                f"{post['caption']} {post['hashtags']}\n"
                f"CTA: {post['cta']}\n"
                f"Tone: {post['tone']}"
            )

            enriched_versions = self.enrich_post(post)
            for j, enriched in enumerate(enriched_versions, 1):
                output.append(f"✨ Enriched Version {j}: {enriched.strip()}")

            structured_post = f"✨ Structured Enriched Post: {enriched_versions[0]}"
            output.append(structured_post)

        return "\n".join(output) if output else "No matching posts found."



generator = SocialPostGenerator("generated_posts.csv")
