# STEP 7: Model Evaluation

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Config
# ==============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_dir = r"C:\Sham\Project\TrashNet\dataset\test"

# ==============================
# 2. Load Test Data
# ==============================
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(test_data.class_indices.keys())

# ==============================
# 3. Load Trained Model
# ==============================
model = load_model("waste_classifier_model_final.keras")

# ==============================
# 4. Evaluate Model
# ==============================
test_loss, test_accuracy = model.evaluate(test_data)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")

# ==============================
# 5. Predictions
# ==============================
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)
true_labels = test_data.classes

# ==============================
# 6. Classification Report
# ==============================
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))

# ==============================
# 7. Confusion Matrix
# ==============================
cm = confusion_matrix(true_labels, pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
