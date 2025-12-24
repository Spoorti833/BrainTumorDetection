import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

# Load trained model
model = load_model('BrainTumor10EpochsCategorical.h5')

# Load image
image = cv2.imread('uploads/no3.jpg')
if image is None:
    print("Image not found")
    exit()

# Preprocess image
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

# 🔹 MODEL PREDICTION
pred = model.predict(input_img, verbose=0)
result = np.argmax(pred, axis=-1)

# 🔹 ADD THESE LINES HERE 👇
classes = ['No Tumor', 'Tumor']
print("Prediction:", classes[result[0]])

confidence = np.max(pred) * 100
print(f"Confidence: {confidence:.2f}%")
