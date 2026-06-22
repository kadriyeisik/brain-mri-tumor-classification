import os
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

models = ["Custom CNN", "EfficientNetB0"]
accuracies = [86.6, 89.6]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies)

plt.ylabel("Validation Accuracy (%)")
plt.title("Model Performance Comparison")
plt.ylim(0, 100)

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.1f}%", ha="center")

plt.savefig("images/model_comparison.png")
plt.close()

print("Chart created successfully!")
epochs = [1, 2, 3, 4, 5]
train_accuracy = [81.89, 88.46, 90.41, 92.48, 93.52]
val_accuracy = [82.50, 85.44, 87.75, 89.56, 89.63]

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_accuracy, marker="o", label="Train Accuracy")
plt.plot(epochs, val_accuracy, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("EfficientNetB0 Training and Validation Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("images/efficientnet_accuracy.png")
plt.close()

print("Accuracy chart created successfully!")
train_loss = [0.4835, 0.3074, 0.2422, 0.2029, 0.1765]
val_loss = [0.4402, 0.4155, 0.3708, 0.3830, 0.3536]

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNetB0 Training and Validation Loss")

plt.legend()
plt.grid(True)

plt.savefig("images/efficientnet_loss.png")
plt.close()

print("Loss chart created successfully!")