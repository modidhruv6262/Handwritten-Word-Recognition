import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pickle
import os

st.set_page_config(page_title="Handwriting Recognizer", layout="centered", page_icon="🖊️")

st.title("🖊️ Handwriting to Text Extractor")
st.write("Select a random image from the dataset to see the model's prediction!")

# ==========================================
# 1. CONSTANTS & PATHS
# ==========================================
MODEL_FILE = "handwriting_prediction_model.keras"
VOCAB_FILE = "vocab.pkl"
SAMPLE_DIR = "sample_images"
IMG_WIDTH = 128
IMG_HEIGHT = 32

# ==========================================
# 2. LOAD MODEL & VOCABULARY
# ==========================================
@st.cache_resource
def load_assets():
    # Load Vocabulary
    if not os.path.exists(VOCAB_FILE):
        st.error(f"❌ '{VOCAB_FILE}' not found!")
        st.stop()
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    
    num_to_char = layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True)
    
    # Load Model
    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ '{MODEL_FILE}' not found!")
        st.stop()
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    
    return model, num_to_char

model, num_to_char = load_assets()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    return tf.expand_dims(img, 0)

def decode_prediction(pred):
    max_length = 64
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :max_length]
    text = tf.strings.reduce_join(num_to_char(results[0])).numpy().decode("utf-8")
    return text.replace("[UNK]", "").strip()

# ==========================================
# 4. UI: IMAGE SELECTOR
# ==========================================
if os.path.exists(SAMPLE_DIR):
    sample_images = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")]
else:
    sample_images = []

if not sample_images:
    st.warning("No sample images found in 'sample_images' folder.")
else:
    # Dropdown menu for images
    selected_img_name = st.selectbox("Choose a sample image:", sample_images)
    
    if selected_img_name:
        img_path = os.path.join(SAMPLE_DIR, selected_img_name)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Selected Image")
            st.image(img_path, use_container_width=True)
            
        with col2:
            st.subheader("Model Prediction")
            with st.spinner("Analyzing..."):
                # Preprocess & Predict
                img_tensor = preprocess_image(img_path)
                pred = model.predict(img_tensor, verbose=0)
                decoded_text = decode_prediction(pred)
                
            st.success(f"**{decoded_text}**")
