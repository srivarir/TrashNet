# STEP 9: Explainability using Occlusion Sensitivity (STABLE METHOD)

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==============================
# Config
# ==============================
IMG_SIZE = (224, 224)
PATCH_SIZE = 32          # size of occlusion patch
STRIDE = 16              # movement step
IMAGE_PATH = r"C:\Sham\Project\TrashNet\metaltubes.jpeg"

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ==============================
# Load model
# ==============================
model = load_model("waste_classifier_model_final.keras")

# ==============================
# Load & preprocess image
# ==============================
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0

input_img = np.expand_dims(img_array, axis=0)

# ==============================
# Get base prediction
# ==============================
base_pred = model.predict(input_img)
target_class = np.argmax(base_pred)
base_confidence = base_pred[0][target_class]

# ==============================
# Occlusion map
# ==============================
heatmap = np.zeros((IMG_SIZE[0], IMG_SIZE[1]))

for y in range(0, IMG_SIZE[0], STRIDE):
    for x in range(0, IMG_SIZE[1], STRIDE):
        occluded = img_array.copy()
        occluded[y:y+PATCH_SIZE, x:x+PATCH_SIZE, :] = 0

        occluded_input = np.expand_dims(occluded, axis=0)
        pred = model.predict(occluded_input, verbose=0)

        confidence_drop = base_confidence - pred[0][target_class]
        heatmap[y:y+PATCH_SIZE, x:x+PATCH_SIZE] = confidence_drop

# Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# ==============================
# Overlay heatmap
# ==============================
original = cv2.imread(IMAGE_PATH)
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)

# ==============================
# Display
# ==============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap_resized, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Prediction: {class_names[target_class]}")
plt.axis("off")

plt.show()
