"""
⚠️ Training Instructions:

This script can be used to train **either one model at a time** or both models sequentially.

1. **Train models separately (recommended)**:
   - For environment classification:
       - Set `train_dir` and `val_dir` to your environment dataset paths.
       - Save the model with `model.save("env_model.keras")`.
   - For haze detection:
       - Set `train_dir` and `val_dir` to your haze dataset paths.
       - Save the model with `model.save("haze_model.keras")`.
   - You can reuse the same script for both — just change the dataset paths and model save name.

2. **Train both models sequentially in one run (optional)**:
   - Duplicate the model definition and training blocks within the same script:
       - First block: environment model
       - Second block: haze model
   - Make sure each model uses its own dataset and save name.
   - Example:
       - `env_model.save("env_model.keras")`
       - `haze_model.save("haze_model.keras")`

📌 Notes:
- Ensure your datasets are organized correctly (subfolders for classes).
- Adjust architectures if needed per task.
- Large `.keras` files should not be committed to GitHub; only commit the script.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import os

# Set dataset paths
base_dir = r"C:\path\to\\dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# 1. Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# 2. CNN Model
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 3. Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True
)

# 5. Train
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# 6. Save
model.save("cnn_model.keras")

# 7. Plot
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Val Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()
