# STEP 6C: Final Accuracy Boost (>85%)

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# ==============================
# Config
# ==============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

train_dir = r"C:\Sham\Project\TrashNet\dataset\train"
val_dir = r"C:\Sham\Project\TrashNet\dataset\val"

# ==============================
# Data Generators
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ==============================
# Compute Class Weights
# ==============================
labels = train_data.classes
class_counts = np.bincount(labels)
class_weights = {i: max(class_counts)/class_counts[i] for i in range(len(class_counts))}

print("Class weights:", class_weights)

# ==============================
# Load Model
# ==============================
model = load_model("waste_classifier_model_finetuned.h5")

base_model = model.layers[1]

# ==============================
# Unfreeze TOP 50 layers
# ==============================
for layer in base_model.layers[-50:]:
    layer.trainable = True

# ==============================
# Recompile (slightly higher LR)
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# Train
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ==============================
# Save Final Model
# ==============================
model.save("waste_classifier_model_final.keras")
