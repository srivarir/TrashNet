# STEP 3: Data Loading & Augmentation (COMPLETE SCRIPT)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ==============================
# 1. Dataset Paths
# ==============================
train_dir = r"C:\Sham\Project\TrashNet\dataset\train"
val_dir = r"C:\Sham\Project\TrashNet\dataset\val"
test_dir = r"C:\Sham\Project\TrashNet\dataset\test"

# ==============================
# 2. Image Parameters
# ==============================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ==============================
# 3. Data Augmentation (Training)
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values
    rotation_range=20,       # Random rotation
    width_shift_range=0.2,   # Horizontal shift
    height_shift_range=0.2,  # Vertical shift
    zoom_range=0.2,          # Zoom
    horizontal_flip=True    # Flip images
)

# ==============================
# 4. Validation & Test Generator
# ==============================
val_test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==============================
# 5. Load Training Data
# ==============================
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ==============================
# 6. Load Validation Data
# ==============================
val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

# ==============================
# 7. Load Test Data
# ==============================
test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==============================
# 8. Display Class Labels
# ==============================
print("Class Indices:")
print(train_data.class_indices)

# ==============================
# 9. Visualize Sample Images
# ==============================
images, labels = next(train_data)

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.axis("off")

plt.suptitle("Sample Training Images")
plt.show()
