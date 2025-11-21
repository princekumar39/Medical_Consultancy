import random
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L12-v2', device="cpu")

# Load the dataset 
df = pd.read_csv('data.csv')  # Ensure 'data.csv' is in the same directory

medical_keywords = {
    "fever": "It sounds like you may have a fever. Stay hydrated and consider seeing a doctor if symptoms persist.",
    "cough": "A persistent cough might be due to an infection or allergy. Try warm fluids and rest.",
    "headache": "Headaches can have many causes, including stress and dehydration. Consider resting and drinking water.",
    "cold": "Common colds usually go away on their own. Stay warm, drink fluids, and get rest.",
}

health_tips = {
    "sleep": [
        "Try to get at least 7-8 hours of sleep each night.",
        "Establish a regular sleep routine to improve sleep quality.",
        "Avoid screens before bed to help your mind relax.",
    ],
    "energy": [
        "Make sure you're eating a balanced diet to maintain energy.",
        "Exercise regularly to boost your energy levels.",
        "Stay hydrated throughout the day to avoid fatigue.",
    ],
    "stress": [
        "Take short breaks throughout the day to reduce stress.",
        "Practice mindfulness or meditation to help manage stress.",
        "Engage in physical activity to reduce anxiety and stress.",
    ],
    "general": [
        "Drink plenty of water throughout the day.",
        "Get at least 30 minutes of exercise every day.",
        "Eat a balanced diet rich in fruits and vegetables.",
    ],
}

# Function to get personalized health tip
def get_personalized_health_tip(user_input):
    user_input_lower = user_input.lower()
    if "tired" in user_input_lower or "fatigue" in user_input_lower:
        return random.choice(health_tips["energy"])
    elif "sleep" in user_input_lower or "rest" in user_input_lower:
        return random.choice(health_tips["sleep"])
    elif "stress" in user_input_lower or "anxious" in user_input_lower:
        return random.choice(health_tips["stress"])
    else:
        return random.choice(health_tips["general"])

# Function to find best cure based on similarity
def find_best_cure(user_input):
    user_input_embedding = model.encode(user_input, convert_to_tensor=True)
    disease_embeddings = model.encode(df['disease'].tolist(), convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(user_input_embedding, disease_embeddings)[0]
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()
    
    SIMILARITY_THRESHOLD = 0.5
    
    if best_match_score < SIMILARITY_THRESHOLD:
        for keyword, response in medical_keywords.items():
            if keyword in user_input.lower():
                return response
        return "I'm sorry, I don't have enough information on this. Please consult a healthcare professional."
    
    return df.iloc[best_match_idx]['cure']

# Function to translate text
def translate_text(text, dest_language='en'):
      try:
        return GoogleTranslator(source='auto', target=dest_language).translate(text)
      except exceptions.NotValidPayload:
        return text
      except exceptions.TranslationNotFound:
        return text
      except Exception:
        # if translator takes too long or fails return original
        return text

# Streamlit UI
st.title("Medical Chatbot 🤖")
user_input = st.text_input("Ask a question:")

language_choice = st.selectbox("Select Language", [
    "English", "Hindi", "Gujarati", "Korean", "Turkish",
    "German", "French", "Arabic", "Urdu", "Tamil", "Telugu", "Chinese", "Japanese"
])

language_codes = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Korean": "ko",
    "Turkish": "tr",
    "German": "de",
    "French": "fr",
    "Arabic": "ar",
    "Urdu": "ur",
    "Tamil": "ta",
    "Telugu": "te",
    "Chinese": "zh-CN",
    "Japanese": "ja",
}

# Response button
if st.button("Get Response"):
    if user_input:
        response = find_best_cure(user_input)
        translated_response = translate_text(response, dest_language=language_codes[language_choice])
        st.write(f"**My Suggestion is:** {translated_response}")
        st.write("*Please note, the translation is provided by AI and might not be perfect.*")

# Personalized health tip button
if st.button("Get a Personalized Health Tip"):
    if user_input:
        personalized_tip = get_personalized_health_tip(user_input)
        translated_tip = translate_text(personalized_tip, dest_language=language_codes[language_choice])
        st.write(f"**Health Tip:** {translated_tip}")
        st.write("*Please note, the translation is provided by AI and might not be perfect.*")
