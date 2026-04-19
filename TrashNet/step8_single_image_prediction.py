# STEP 8: Single Image Prediction

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ==============================
# 1. Config
# ==============================
IMG_SIZE = (224, 224)

# ⚠️ Change this to your test image path
IMAGE_PATH = "C:\Sham\Project\TrashNet\metaltubes.jpeg"

# ==============================
# 2. Class Labels (same order as training)
# ==============================
class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ==============================
# 3. Load Trained Model
# ==============================
model = load_model("waste_classifier_model_final.keras")

# ==============================
# 4. Load & Preprocess Image
# ==============================
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0          # normalize
img_array = np.expand_dims(img_array, axis=0)

# ==============================
# 5. Predict
# ==============================
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
confidence = np.max(predictions)

# ==============================
# 6. Output
# ==============================
print("Predicted Class:", class_names[predicted_class])
print(f"Confidence: {confidence * 100:.2f}%")

# ==============================
# 7. Display Image
# ==============================
plt.imshow(img)
plt.axis("off")
plt.title(f"{class_names[predicted_class]} ({confidence*100:.2f}%)")
plt.show()
