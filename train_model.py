import tensorflow as tf

training_path = "dataset/Training"
testing_path = "dataset/Testing"

img_size = (224, 224)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    training_path,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale"
)

print("Sınıf isimleri:")
print(train_data.class_names)
for images, labels in train_data.take(1):
    print("Images shape:")
    print(images.shape)

    print("Labels shape:")
    print(labels.shape)

    print("Labels:")
    print(labels.numpy())
    model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 1)),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()
model.save("model/brain_tumor_model.keras")

print("Model kaydedildi.")