import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

model = tf.keras.models.load_model("model/brain_tumor_model.keras")

testing_path = "dataset/Testing"
img_size = (224, 224)
batch_size = 32

test_data = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    shuffle=False
)

class_names = test_data.class_names

y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()