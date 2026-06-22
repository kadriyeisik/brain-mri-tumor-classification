import time

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="Brain MRI Tumor Classification",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
    color: white;
}

h1, h2, h3 {
    color: #60a5fa !important;
}

[data-testid="stSidebar"] {
    background-color: #1e293b;
}

[data-testid="stMetricValue"] {
    color: #38bdf8;
}

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_brain_model():
    return tf.keras.models.load_model("model/brain_tumor_efficientnet.keras")


model = load_brain_model()

class_names = ["glioma", "meningioma", "notumor", "pituitary"]


st.sidebar.title("🧠 Project Info")

st.sidebar.markdown("""
### Model
EfficientNetB0 Transfer Learning

### Performance
Validation Accuracy: **89.6%**

### Dataset
Training Images: **5600**  
Testing Images: **1600**

### Classes
- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor
""")

st.sidebar.warning(
    "Educational use only. Not for medical diagnosis."
)


st.title("🧠 Brain MRI Tumor Classification Dashboard")

st.markdown("""
This application uses an **EfficientNetB0 Transfer Learning** model to classify brain MRI images into four categories:

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**
""")

st.divider()


uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded MRI")
        st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    start_time = time.time()
    predictions = model.predict(img_array, verbose=0)
    end_time = time.time()

    prediction_time = end_time - start_time

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    with col2:
        st.subheader("Prediction Result")
        st.metric("Predicted Class", predicted_class.upper())
        st.metric("Confidence", f"{confidence:.2f}%")
        st.metric("Prediction Time", f"{prediction_time:.3f} sec")

    st.divider()

    st.subheader("Class Probabilities")

    for class_name, probability in zip(class_names, predictions[0]):
        st.write(f"**{class_name.capitalize()}**: {probability * 100:.2f}%")
        st.progress(float(probability))

else:
    st.info("Please upload an MRI image to get a prediction.")


st.divider()

st.header("📊 Model Performance Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "Model Comparison",
    "Accuracy",
    "Loss",
    "Confusion Matrix"
])

with tab1:
    st.image("images/model_comparison.png", use_container_width=True)

with tab2:
    st.image("images/efficientnet_accuracy.png", use_container_width=True)

with tab3:
    st.image("images/efficientnet_loss.png", use_container_width=True)

with tab4:
    st.image("images/confusion_matrix.png", use_container_width=True)


st.divider()

st.warning(
    "Disclaimer: This application is developed for educational and research purposes only. "
    "It should not be used for medical diagnosis or treatment decisions."
)