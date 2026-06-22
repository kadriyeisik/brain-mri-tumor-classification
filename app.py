import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

model = tf.keras.models.load_model("model/brain_tumor_efficientnet.keras")

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

st.title("🧠 Brain MRI Tumor Classification")
st.markdown("""
This application uses a **Convolutional Neural Network (CNN)** model to classify brain MRI images into four categories:

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor
""")

st.divider()

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    with col2:
        st.subheader("Prediction Result")
        st.metric("Predicted Class", predicted_class.upper())
        st.metric("Confidence", f"{confidence:.2f}%")

    st.divider()

    st.subheader("Class Probabilities")

    for class_name, probability in zip(class_names, predictions[0]):
        st.progress(float(probability))
        st.write(f"**{class_name.capitalize()}**: {probability * 100:.2f}%")

else:
    st.info("Please upload an MRI image to get a prediction.")

st.divider()

st.warning(
    "Disclaimer: This application is developed for educational purposes only "
    "and should not be used for medical diagnosis."
)