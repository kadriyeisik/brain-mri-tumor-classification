import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model/brain_tumor_model.keras")

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

test_images = {
    "glioma": "dataset/Testing/glioma/Te-gl_1.jpg",
    "meningioma": "dataset/Testing/meningioma/Te-me_1.jpg",
    "notumor": "dataset/Testing/notumor/Te-no_1.jpg",
    "pituitary": "dataset/Testing/pituitary/Te-pi_1.jpg"
}

for actual_class, image_path in test_images.items():
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224),
        color_mode="grayscale"
    )

    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    print("----------------------------")
    print("Actual class:", actual_class)
    print("Predicted class:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")
    print("All probabilities:", predictions[0])