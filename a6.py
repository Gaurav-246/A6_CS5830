from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from tensorflow.keras.models import Sequential
import keras
import numpy as np
import sys
import argparse


# Task 1 - Q1
app = FastAPI()

# Task 1 - Q2
model_path = sys.argv[1]        # Model path from terminal

# Task 1 - Q3
def load_model_weights(model_path : str):
    model = keras.models.load_model(model_path)
    return model

# Task 1 - Q4
def predict_digit(model:Sequential, data_point:list):
    data_point = np.array(data_point) # 784 points
    data_point = data_point / 255.0   # Normalize for prediction
    pred = model.predict(data_point)
    return str(np.argmax(pred))

# Task 1 - Q5
@app.post('/predict')
async def predict(upload_file: UploadFile):
    contents = await upload_file.read()         # Read the file in bytes
    data_point = [int(byte) for byte in contents]   # Convert it to a serialized array of length 784
    model = load_model_weights(model_path)  
    digit = predict_digit(model, data_point)
    return {"digit": digit}
    

