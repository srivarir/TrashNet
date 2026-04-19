# STEP 9: Grad-CAM (STABLE & GUARANTEED VERSION)

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

# ==============================
# Config
# ==============================
IMG_SIZE = (224, 224)
IMAGE_PATH = r"C:\Sham\Project\TrashNet\metaltubes.jpeg"
WEIGHTS_PATH = "waste_classifier_model_final.keras"

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ==============================
# 1. Rebuild model architecture
# ==============================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(6, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ==============================
# 2. Load trained weights
# ==============================
model.load_weights(WEIGHTS_PATH)

# ==============================
# 3. Load & preprocess image
# ==============================
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# ==============================
# 4. Build Grad-CAM model
# ==============================
last_conv_layer = base_model.get_layer("Conv_1")

grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[last_conv_layer.output, model.output]
)

# ==============================
# 5. Compute Grad-CAM
# ==============================
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_array)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

conv_outputs = conv_outputs[0]
heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
heatmap = tf.maximum(heatmap, 0)
heatmap /= tf.reduce_max(heatmap)
heatmap = heatmap.numpy()

# ==============================
# 6. Overlay heatmap
# ==============================
img_original = cv2.imread(IMAGE_PATH)
img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

heatmap = cv2.resize(heatmap, (img_original.shape[1], img_original.shape[0]))
heatmap_colored = cv2.applyColorMap(
    np.uint8(255 * heatmap),
    cv2.COLORMAP_JET
)

overlay = cv2.addWeighted(img_original, 0.6, heatmap_colored, 0.4, 0)

# ==============================
# 7. Display
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
plt.imshow(overlay)
plt.title(f"Prediction: {class_names[class_idx]}")
plt.axis("off")

plt.show()
