# STEP 9: Grad-CAM (FINAL FIXED VERSION FOR MobileNetV2)

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ==============================
# Config
# ==============================
IMG_SIZE = (224, 224)
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
img_array = np.expand_dims(img_array, axis=0)

# ==============================
# Get LAST convolution layer properly
# ==============================
base_model = model.get_layer("mobilenetv2_1.00_224")
last_conv_layer = base_model.get_layer("Conv_1")

# Build Grad-CAM model
grad_model = tf.keras.models.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

# ==============================
# Compute gradients
# ==============================
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_index = tf.argmax(predictions[0])
    loss = predictions[:, class_index]

grads = tape.gradient(loss, conv_outputs)

# ==============================
# Generate heatmap
# ==============================
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# ==============================
# Overlay heatmap
# ==============================
img_original = cv2.imread(IMAGE_PATH)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

heatmap_resized = cv2.resize(
    heatmap, (img_original.shape[1], img_original.shape[0])
)

heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap_resized),
    cv2.COLORMAP_JET
)

superimposed_img = cv2.addWeighted(
    img_original, 0.6, heatmap_colored, 0.4, 0
)

# ==============================
# Display
# ==============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img_original)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(superimposed_img)
plt.title(f"Prediction: {class_names[class_index]}")
plt.axis("off")

plt.show()
