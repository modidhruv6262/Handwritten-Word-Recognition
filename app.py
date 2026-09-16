import os
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

st.set_page_config(page_title="Handwritten Word Recognition", page_icon="✍️", layout="wide")

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>✍️ Handwritten Word Recognition</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #7F8C8D;'>An End-to-End Deep Learning OCR System</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

IMG_WIDTH  = 128
IMG_HEIGHT = 32
MAX_LENGTH = 64
VOCAB_FILE = "vocab.pkl"
SAMPLE_DIR = "sample_images"

@st.cache_resource
def load_vocab():
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    num_to_char = layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True)
    return vocab, num_to_char

@st.cache_resource
def build_model(vocab_size):
    inp = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image", dtype="float32")
    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Reshape(((IMG_WIDTH // 4), (IMG_HEIGHT // 4) * 64))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64,  return_sequences=True, dropout=0.25))(x)
    out = layers.Dense(vocab_size + 1, activation="softmax", name="dense2")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.load_weights("model_weights.weights.h5")
    return model

def preprocess(img_bytes):
    img = tf.image.decode_png(img_bytes, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    return tf.expand_dims(img, 0)

def ctc_decode(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :MAX_LENGTH]
    text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode("utf-8")
    return text.replace("[UNK]", "").strip()

col_info, col_data, col_metrics = st.columns(3, gap="large")

with col_info:
    with st.expander("📊 Project Overview", expanded=False):
        st.info(
            "This application is the deployment phase of a complete deep learning pipeline designed to "
            "read and digitize handwritten words from scanned or photographed images.\n\n"
            "It is useful for digitizing handwritten notes, forms, documents, and any real-world "
            "handwriting into machine-readable text automatically."
        )

with col_data:
    with st.expander("🧬 Model Architecture", expanded=False):
        st.success(
            "The model is a **CRNN (Convolutional Recurrent Neural Network)** combining two technologies:\n\n"
            "* **CNN Layers:** Act as the eyes — they scan the image and extract visual features like loops, lines, and curves.\n"
            "* **Bidirectional LSTM Layers:** Act as the brain — they read the features left-to-right AND right-to-left to understand the word as a sequence."
        )

with col_metrics:
    with st.expander("⚙️ Training & Dataset", expanded=False):
        st.warning(
            "Trained on the **IAM Handwriting Word Database** — the industry-standard benchmark for English handwriting recognition containing over **115,000 word images** from **657 unique writers**.\n\n"
            "* **Loss Function:** CTC (Connectionist Temporal Classification) — allows the model to align variable-length images to variable-length words without needing character-level annotation."
        )

st.markdown("<br><hr><br>", unsafe_allow_html=True)

spacer_left, content_col, spacer_right = st.columns([1, 2, 1])

with content_col:
    st.markdown("<h3 style='text-align: center;'>🖼️ Live Prediction</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Select an image from the dataset below and the model will predict the handwritten word in real time.</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    missing = []
    if not os.path.exists(VOCAB_FILE):
        missing.append(f"`{VOCAB_FILE}`")
    if not os.path.exists("model_weights.weights.h5"):
        missing.append("`model_weights.weights.h5`")
    if missing:
        st.error(f"Missing files: {', '.join(missing)}. Please add them to your repo.")
        st.stop()

    vocab, num_to_char = load_vocab()
    model = build_model(len(vocab))

    if os.path.exists(SAMPLE_DIR):
        images = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")])
    else:
        images = []

    if not images:
        st.warning(f"No PNG images found in `{SAMPLE_DIR}/` folder.")
        st.stop()

    selected = st.selectbox("🖼️ Choose an image:", images)

    if selected:
        img_path = os.path.join(SAMPLE_DIR, selected)

        pred_col1, pred_col2 = st.columns(2)

        with pred_col1:
            st.markdown("**Input Image**")
            st.image(img_path, use_container_width=True)

        with pred_col2:
            st.markdown("**Prediction**")
            with st.spinner("Running model..."):
                raw = tf.io.read_file(img_path)
                tensor = preprocess(raw)
                pred = model.predict(tensor, verbose=0)
                result = ctc_decode(pred, num_to_char)

            if result:
                st.success(f"### {result}")
            else:
                st.warning("Model returned an empty prediction.")

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<hr style="border-color: rgba(255, 255, 255, 0.1); margin-bottom: 2rem;">
<div style="text-align: center;">
    <h3 style="margin-bottom: 5px;">Dhruv Modi</h3>
    <p style="color: #7F8C8D; font-size: 16px; margin-bottom: 20px;">Data Science Student & Developer</p>
    <div style="display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
        <a href="mailto:dhruvmodi602@gmail.com" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" alt="Email">
        </a>
        <a href="https://github.com/modidhruv6262" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/dhruvmodi6262" target="_blank" style="text-decoration: none;">
            <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
        </a>
    </div>
</div>
<br>
""", unsafe_allow_html=True)
