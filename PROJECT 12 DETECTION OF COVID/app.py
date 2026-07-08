import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-ray Detection",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 COVID-19 Detection from Chest X-ray")
st.write("Upload a Chest X-ray image to predict whether it is **COVID-19** or **Normal**.")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("model.keras")

model = load_my_model()

IMG_SIZE = (299, 299)

# ----------------------------
# Upload Image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    probability = float(prediction[0][0])

    st.subheader("Prediction")

    if probability > 0.5:
        st.error("🦠 COVID-19 Detected")
        st.write(f"Confidence: **{probability*100:.2f}%**")
    else:
        st.success("✅ Normal")
        st.write(f"Confidence: **{(1-probability)*100:.2f}%**")
