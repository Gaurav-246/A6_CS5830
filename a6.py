from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import Sequential
import tensorflow as tf
import uvicorn
import keras
import numpy as np
import os
import io
import matplotlib.pyplot as plt
import cv2

# Task 1 - Q1
app = FastAPI()

# Task 1 - Q2
model_path = str(os.getenv("MODEL_PATH"))  # Model path from terminal
# Run "export MODEL_PATH=/path/to/model" in terminal


# Task 1 - Q3
def load_model_weights(model_path: str):
    model = tf.keras.models.load_model(model_path)
    return model


# Task 1 - Q4
def predict_digit(model: Sequential, data_point: list):
    data_point = np.array(data_point)  # 784 points
    data_point = np.expand_dims(data_point, axis=0)
    pred = model.predict(data_point)
    return str(np.argmax(pred))


# Task 2 - Q1,Q2,Q3


def format_image(contents, size=28, interpolation=cv2.INTER_AREA):
    with io.BytesIO(contents) as f:
        img = plt.imread(f)
    img = img[:,:,:3]
    resized = cv2.resize(
        img, (size, size), interpolation=interpolation
    )  # Resize the image
    resized = cv2.bitwise_not(resized)
    data = np.array(resized)
    data = np.mean(data, axis=2) # Convert to grayscale
    data = data.flatten()  # Serialize to an array of 784 points
    data = data / 255.0  # Normalize
    return data

# Task 1 - Q5
@app.post("/predict")
async def predict(my_file: UploadFile):
    contents = await my_file.read()  # Read the file in bytes
    data_point = format_image(contents)
    model = load_model_weights(model_path)
    digit = predict_digit(model, data_point)
    return {"digit": digit}
