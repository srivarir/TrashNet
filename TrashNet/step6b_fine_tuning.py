# STEP 6B: Fine-Tuning to push accuracy >85%

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ==============================
# 1. Config
# ==============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 10

train_dir = r"C:\Sham\Project\TrashNet\dataset\train"
val_dir = r"C:\Sham\Project\TrashNet\dataset\val"

# ==============================
# 2. Data Generators (same as before)
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
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
# 3. Load Previously Trained Model
# ==============================
model = load_model("waste_classifier_model.h5")

# ==============================
# 4. Unfreeze TOP layers of MobileNetV2
# ==============================
base_model = model.layers[1]   # MobileNetV2

# Unfreeze last 20 layers
for layer in base_model.layers[-20:]:
    layer.trainable = True

# ==============================
# 5. Recompile with LOW learning rate
# ==============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==============================
# 6. Fine-tune
# ==============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS
)

# ==============================
# 7. Save improved model
# ==============================
model.save("waste_classifier_model_finetuned.h5")

# ==============================
# 8. Plot Accuracy
# ==============================
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Fine-tuning Accuracy")
plt.show()
