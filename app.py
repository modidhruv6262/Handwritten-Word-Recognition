import os
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

st.set_page_config(
    page_title="Handwritten Word Recognition",
    page_icon="✍️",
    layout="centered",
)

IMG_WIDTH   = 128
IMG_HEIGHT  = 32
MAX_LENGTH  = 64
VOCAB_FILE  = "vocab.pkl"
SAMPLE_DIR  = "sample_images"

@st.cache_resource
def load_vocab():
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    num_to_char = layers.StringLookup(
        vocabulary=vocab, mask_token=None, invert=True
    )
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
    decoded = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :MAX_LENGTH]
    text = tf.strings.reduce_join(num_to_char(decoded[0])).numpy().decode("utf-8")
    return text.replace("[UNK]", "").strip()

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

st.title("✍️ Handwritten Word Recognition")
st.write("Select an image from the dataset below and the model will predict the word.")
st.divider()

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
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Image**")
        st.image(img_path, use_container_width=True)

    with col2:
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
