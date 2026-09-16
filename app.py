import os
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Handwritten Word Recognition",
    page_icon="✍️",
    layout="centered",
)

# ── Constants ────────────────────────────────────────────────────
IMG_WIDTH   = 128
IMG_HEIGHT  = 32
MAX_LENGTH  = 64
VOCAB_FILE  = "vocab.pkl"
SAMPLE_DIR  = "sample_images"

<<<<<<< HEAD
# ── Load vocabulary ──────────────────────────────────────────────
@st.cache_resource
def load_vocab():
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    num_to_char = layers.StringLookup(
        vocabulary=vocab, mask_token=None, invert=True
    )
    return vocab, num_to_char
=======
MODEL_FILE = "handwriting_prediction_model.keras"
VOCAB_FILE = "vocab.pkl"
SAMPLE_DIR = "sample_images"
IMG_WIDTH = 128
IMG_HEIGHT = 32

class CTCLayer(layers.Layer):
    def call(self, y_true, y_pred, label_lengths):
        batch_len = tf.cast(tf.shape(y_true)[0], "int64")
        input_length = tf.cast(tf.shape(y_pred)[1], "int64") * tf.ones((batch_len, 1), dtype="int64")
        label_length = tf.cast(tf.reshape(label_lengths, (-1, 1)), "int64")
        self.add_loss(tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length))
        return y_pred

@st.cache_resource
def load_assets():
    if not os.path.exists(VOCAB_FILE):
        st.error(f"❌ '{VOCAB_FILE}' not found!")
        st.stop()
    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)

    num_to_char = layers.StringLookup(vocabulary=vocab, mask_token=None, invert=True)

    if not os.path.exists(MODEL_FILE):
        st.error(f"❌ '{MODEL_FILE}' not found!")
        st.stop()
    model = tf.keras.models.load_model(
        MODEL_FILE,
        custom_objects={"CTCLayer": CTCLayer},
        compile=False
    )
    return model, num_to_char
>>>>>>> 4a5f39886f1ac1d4ef46e1136921072f3b42eed9

# ── Build model from scratch + load weights ──────────────────────
@st.cache_resource
def build_model(vocab_size):
    inp = layers.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1), name="image", dtype="float32")

<<<<<<< HEAD
    x = layers.Conv2D(32, (3, 3), activation="relu",
                      kernel_initializer="he_normal", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu",
                      kernel_initializer="he_normal", padding="same")(x)
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

# ── Preprocess image ─────────────────────────────────────────────
def preprocess(img_bytes):
    img = tf.image.decode_png(img_bytes, channels=1)
=======
def preprocess_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
>>>>>>> 4a5f39886f1ac1d4ef46e1136921072f3b42eed9
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.transpose(img, perm=[1, 0, 2])
    return tf.expand_dims(img, 0)

# ── CTC decode ───────────────────────────────────────────────────
def ctc_decode(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    decoded = tf.keras.backend.ctc_decode(
        pred, input_length=input_len, greedy=True
    )[0][0][:, :MAX_LENGTH]
    text = tf.strings.reduce_join(
        num_to_char(decoded[0])
    ).numpy().decode("utf-8")
    return text.replace("[UNK]", "").strip()

<<<<<<< HEAD
# ── Guard: check required files exist ────────────────────────────
missing = []
if not os.path.exists(VOCAB_FILE):
    missing.append(f"`{VOCAB_FILE}`")
if not os.path.exists("model_weights.weights.h5"):
    missing.append("`model_weights.weights.h5`")
if missing:
    st.error(f"Missing files: {', '.join(missing)}. Please add them to your repo.")
    st.stop()
=======
if os.path.exists(SAMPLE_DIR):
    sample_images = [f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")]
else:
    sample_images = []
>>>>>>> 4a5f39886f1ac1d4ef46e1136921072f3b42eed9

vocab, num_to_char = load_vocab()
model = build_model(len(vocab))

# ── UI ───────────────────────────────────────────────────────────
st.title("✍️ Handwritten Word Recognition")
st.write("Select an image from the dataset below and the model will predict the word.")
st.divider()

# Get sample images
if os.path.exists(SAMPLE_DIR):
    images = sorted([f for f in os.listdir(SAMPLE_DIR) if f.endswith(".png")])
else:
<<<<<<< HEAD
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
=======
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
                img_tensor = preprocess_image(img_path)
                pred = model.predict(img_tensor, verbose=0)
                decoded_text = decode_prediction(pred)

            st.success(f"**{decoded_text}**")
>>>>>>> 4a5f39886f1ac1d4ef46e1136921072f3b42eed9
