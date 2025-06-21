import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from deep_translator import GoogleTranslator
from gtts import gTTS

# Load Model
model_path = "s:/New folder/New Plant Diseases Dataset(Augmented)/trained_plant_disease_model.keras"
if not os.path.exists(model_path):
    st.error(f"⚠️ Model file not found at {model_path}! Please check the path.")
    st.stop()

model = tf.keras.models.load_model(model_path)

# Class Labels
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Translator
def translate_text(text, dest_language='en'):
    try:
        return GoogleTranslator(source='auto', target=dest_language).translate(text)
    except Exception as e:
        return f"⚠️ Translation failed: {str(e)}"

# Prediction Function
def predict_disease(image):
    try:
        img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Get confidence percentage
        return class_labels[class_index], confidence
    except Exception as e:
        return f"Prediction failed: {str(e)}", None

# Leaf Health Index Score
def calculate_health_score(confidence, disease):
    if "healthy" in disease.lower():
        return 100, "✅ Healthy", "green"
    elif confidence > 90:
        return 40, "🔴 Critical", "red"
    elif confidence > 70:
        return 60, "🟡 Moderate", "orange"
    else:
        return 80, "🟢 Mild", "blue"

# Text-to-Speech Function
def text_to_speech(text, lang='en'):
    tts = gTTS(text, lang=lang)
    tts.save("output.mp3")
    return "output.mp3"

# Streamlit UI with Updated Bright Design
st.sidebar.title("🌿 Common Treatment")
page = st.sidebar.radio("📌 Navigation", ["🏠 Disease Detection", "💊 Treatment Guide"])

# Logo Handling
logo_path = "logo.jpg"
if os.path.exists(logo_path):
    st.image(logo_path, width=200)
else:
    st.warning("⚠️ Logo not found! Please place 'logo.png' in the project directory.")

if page == "🏠 Disease Detection":
    st.markdown("""
        <h1 style='text-align: center; color: #228B22;'>LeafVIVE</h1>
        <h3 style='text-align: center; color: #4CAF50;'>Detecting Disease & Protecting Green!</h3>
        <hr style='border: 1px solid #228B22;'>
    """, unsafe_allow_html=True)

    # Upload or Capture Image
    uploaded_file = st.file_uploader("📸 Upload Plant Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="🌱 Uploaded Image", use_column_width=True)
        predicted_disease, confidence = predict_disease(uploaded_file)

        if confidence:
            st.success(f"**Predicted Disease:** {predicted_disease} ({confidence:.2f}% confidence)")
            leaf_health_score, health_status, color = calculate_health_score(confidence, predicted_disease)
            st.subheader("🌿 Leaf Health Index Score")
            st.progress(leaf_health_score)
            st.markdown(f"<h3 style='color:{color}'>🌱 Health Status: {health_status}</h3>", unsafe_allow_html=True)
            
            # Text-to-Speech Output
            if st.button("🔊 Hear Diagnosis"):
                audio_file = text_to_speech(predicted_disease, "en")
                st.audio(audio_file, format="audio/mp3")
        else:
            st.error(predicted_disease)
        
        # Translation Support
        selected_language = st.selectbox("🌍 Select Language", ["en", "hi", "ml", "ta", "te", "kn", "gu", "mr", "bn", "pa", "es", "fr", "zh"])
        translated_disease = translate_text(predicted_disease, selected_language)
        st.info(f"**Translated Disease:** {translated_disease}")

elif page == "💊 Treatment Guide":
    st.markdown("""
        <h1 style='text-align: center; color: #228B22;'>💊 Common Treatment Guide</h1>
        <hr style='border: 1px solid #228B22;'>
    """, unsafe_allow_html=True)
    
    st.subheader("🌾 General Plant Care")
    st.markdown("- Maintain proper watering and soil conditions.")
    st.markdown("- Regularly inspect plants to detect early signs of disease.")
    st.markdown("- Use appropriate pesticides or organic treatments based on disease.")
    
    st.subheader("🍃 Organic Treatment Options")
    st.markdown("- Neem oil spray for fungal infections.")
    st.markdown("- Baking soda solution for powdery mildew.")
    st.markdown("- Compost and natural fertilizers for nutrient enrichment.")

