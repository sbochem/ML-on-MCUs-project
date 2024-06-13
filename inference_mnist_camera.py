import numpy as np
import sys
import serial
import time
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image, ImageTk
import customtkinter as ctk
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2

def capture_and_preprocess_image_cnn():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Cannot capture image from webcam")

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assume largest contour is the number
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # Crop and resize the image around the number
        cropped = gray[y:y+h, x:x+w]
        resized = cv2.resize(cropped, (28, 28))

    else:
        resized = cv2.resize(gray, (28, 28))  # Default resizing if no contours

    # Normalize the image
    normalized_resized = resized / 255.0

    # Reshape the 28x28 image to 28x28x1
    return normalized_resized.reshape(1, 28, 28, 1)


# Prepare the image from webcam
input_data = capture_and_preprocess_image_cnn()

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='classification.tflite')
interpreter.allocate_tensors()

# Get input details
input_details = interpreter.get_input_details()
input_scale, input_zero_point = input_details[0]['quantization']

# Quantize the webcam image
quantized_input_data = np.round(input_data / input_scale + input_zero_point).astype(input_details[0]['dtype'])

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], quantized_input_data)

# Run the model
interpreter.invoke()

# Extract and process the output
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Assuming output also uses quantization
output_scale, output_zero_point = output_details[0]["quantization"]
dequantized_output = (output_data - output_zero_point) * output_scale
predicted_class = np.argmax(output_data)

# Serial communication setup
ser = serial.Serial(port='COM9', baudrate=115200, timeout=3)
ser.flush()
ser.flushInput()
ser.flushOutput()
# Send the quantized image to the MCU
ser.write(quantized_input_data.tobytes())
time.sleep(1)  # Wait for the MCU to process
img = ser.read(28*28)
img = np.frombuffer(img, dtype=np.int8)

input_image_to_plot = quantized_input_data.reshape(28, 28)  # Adjust as per the actual shape

# Normalize img for display if necessary (depends on the data range you expect)
received_image_to_plot = img.reshape(28, 28)

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.suptitle('Comparison of Sent and Received Images')

# Plot the sent image
axes[0].imshow(input_image_to_plot, cmap='gray')
axes[0].title.set_text('Sent Image')
axes[0].axis('off')  # Hide the axes

# Plot the received image
axes[1].imshow(received_image_to_plot, cmap='gray')
axes[1].title.set_text('Received Image')
axes[1].axis('off')  # Hide the axes

# Show the plot
plt.show()

# print("Image received from the MCUs: \n {}".format(img))
time.sleep(1)
pred = ser.read(19)  # Read one byte, which should be the predicted digit
# predicted_digit = pred.decode('utf-8')  # Decode byte to string
decoded_string = pred.decode('utf-8')
digit = None
for char in decoded_string:
    if char.isdigit():
        digit = char
        break

# Method 2: Using regular expressions
import re
match = re.search(r'\d+', decoded_string)
if match:
    digit = match.group(0)

# predicted_class = int(predicted_digit)  # Convert string to integer
print(f"Prediction (from MCU): {digit}")



