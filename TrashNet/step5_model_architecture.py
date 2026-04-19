# STEP 5: Model Architecture (Classifier Head)

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Model

# ==============================
# 1. Basic Config
# ==============================
IMG_SIZE = (224, 224, 3)
NUM_CLASSES = 6   # cardboard, glass, metal, paper, plastic, trash

# ==============================
# 2. Load Pretrained Base Model
# ==============================
base_model = MobileNetV2(
    input_shape=IMG_SIZE,
    include_top=False,
    weights="imagenet"
)

# Freeze base model
base_model.trainable = False

# ==============================
# 3. Build Custom Classifier
# ==============================
inputs = Input(shape=IMG_SIZE)

# Feature extraction
x = base_model(inputs, training=False)

# Reduce feature maps
x = GlobalAveragePooling2D()(x)

# Fully connected layer
x = Dense(128, activation="relu")(x)

# Dropout to reduce overfitting
x = Dropout(0.5)(x)

# Output layer (Softmax for multi-class)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

# ==============================
# 4. Final Model
# ==============================
model = Model(inputs, outputs)

# ==============================
# 5. Model Summary
# ==============================
model.summary()
