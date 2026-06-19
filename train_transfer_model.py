import tensorflow as tf

training_path = "dataset/Training"
testing_path = "dataset/Testing"

img_size = (224, 224)
batch_size = 32

train_data = tf.keras.utils.image_dataset_from_directory(
    training_path,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

test_data = tf.keras.utils.image_dataset_from_directory(
    testing_path,
    image_size=img_size,
    batch_size=batch_size,
    color_mode="rgb"
)

print("Class names:")
print(train_data.class_names)

for images, labels in train_data.take(1):
    print("Images shape:")
    print(images.shape)

    print("Labels shape:")
    print(labels.shape)
    from tensorflow.keras.applications import EfficientNetB0
    
from tensorflow.keras.applications import EfficientNetB0

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

print("Base model loaded successfully!")

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
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