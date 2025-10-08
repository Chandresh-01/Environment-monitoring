"""
Model Tester GUI 🖼️🤖

This script provides a simple interface to test your trained CNN models (environment classification or haze detection).

✅ Features:
- Upload an image through a file dialog.
- Display the image in the GUI.
- Predict the class using the selected trained model.
- Show prediction confidence or probabilities.

💡 Notes:
- Change the model file and class labels to switch between models.
- This script is for testing/prediction only; it does NOT train any models.
- Useful for quickly verifying your model’s performance on individual images.
"""

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load your haze detection model
model = load_model("haze_cnn_model.keras")

# Binary class labels
class_labels = {0: "Clear", 1: "Hazy"}

# GUI Window
root = tk.Tk()
root.title("Haze Detection Classifier")
root.geometry("600x500")
root.configure(bg="white")

# Functions
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display image
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        # Prepare image for model
        img_for_model = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img_for_model) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0][0]
        class_index = int(prediction > 0.5)
        confidence = prediction * 100 if class_index == 1 else (1 - prediction) * 100

        result = f"Class: {class_labels[class_index]}\nConfidence: {confidence:.2f}%"
        result_label.config(text=result)

# Widgets
panel = tk.Label(root)
panel.pack(pady=60)

btn = tk.Button(root, text="Upload Image", command=load_image, font=("Arial", 14), bg="#4CAF50", fg="white")
btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 16), bg="white")
result_label.pack(pady=20)

# Launch GUI
root.mainloop()
