# STEP 4: Model Selection using Transfer Learning (MobileNetV2)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# ==============================
# 1. Image Input Shape
# ==============================
IMG_SIZE = (224, 224, 3)

# ==============================
# 2. Load Pretrained MobileNetV2
# ==============================
base_model = MobileNetV2(
    input_shape=IMG_SIZE,
    include_top=False,      # Remove original classifier
    weights="imagenet"      # Use pretrained ImageNet weights
)

# ==============================
# 3. Freeze Base Model Layers
# ==============================
base_model.trainable = False

# ==============================
# 4. Create Model Input
# ==============================
inputs = Input(shape=IMG_SIZE)

# ==============================
# 5. Pass Input Through Base Model
# ==============================
x = base_model(inputs, training=False)

# ==============================
# 6. Create Feature Extractor Model
# ==============================
model = Model(inputs, x)

# ==============================
# 7. Print Model Summary
# ==============================
model.summary()
