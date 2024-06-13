import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import serial
import time
import cv2

def capture_and_preprocess_image_combined():
    cap = cv2.VideoCapture(0)
    
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 75) 
    # cap.set(cv2.CAP_PROP_CONTRAST, 21) 
    # cap.set(cv2.CAP_PROP_SATURATION, 66) 
    # cap.set(cv2.CAP_PROP_HUE, 50)
    
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Cannot capture image from webcam")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frame = frame.copy()

    # Noise reduction and grayscale
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    
    # Thresholding to find contours
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Adjust to square
        side_length = max(w, h)
        center_x = x + w // 2
        center_y = y + h // 2
        square_x = max(center_x - side_length // 2, 0)
        square_y = max(center_y - side_length // 2, 0)
        square_x_end = min(square_x + side_length, frame.shape[1])
        square_y_end = min(square_y + side_length, frame.shape[0])
        
        # Crop to the square
        cropped_frame = frame[square_y:square_y_end, square_x:square_x_end]

        # Resize to fit the model input
        resized_frame = cv2.resize(cropped_frame, (32, 32), interpolation=cv2.INTER_AREA)
    else:
        resized_frame = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_AREA)

    normalized_resized_frame = resized_frame.astype('float32') / 255.0

    return original_frame, normalized_resized_frame


def main():
    # Load the TensorFlow Lite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path='models/fmnist_full_quant_adapted.tflite')
    interpreter.allocate_tensors()

    # Get input details and quantization parameters
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    x_test = np.load('x_test_cifar.npy')
    
    image_index = 4
    test_image_cifar = x_test[image_index]

    # Example of how to use this function
    original_img, test_image = capture_and_preprocess_image_combined()
    # Quantize the image for the TFLite model
    quantized_image = np.array(test_image / input_scale + input_zero_point, dtype=np.uint8)

    # Local inference
    interpreter.set_tensor(input_details[0]['index'], [quantized_image])
    interpreter.invoke()
    local_output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Local predictions:{local_output}")
    local_predicted_class = np.argmax(local_output)
     # Initialize serial communication
    ser = serial.Serial(port='COM9', baudrate=115200, timeout=10)
    ser.flushInput()
    ser.flushOutput()

    # Send the image to the MCU
    # ser.write(quantized_image.tobytes())
    ser.write(quantized_image.flatten())
    time.sleep(3)  # Allow time for MCU to process the image

    # Read the image back from the MCU
    received_image_data = ser.read(32 * 32 * 3)  # Assuming 3 bytes per pixel for RGB
    received_image = np.frombuffer(received_image_data, dtype=np.uint8).reshape(32, 32, 3)

    # Display both images for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(received_image)
    plt.title('Received Image from MCU')
    plt.show()

    time.sleep(1)
    # Read the prediction from the MCU
    start_time = time.perf_counter()
    pred = ser.read(10)    
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000  # Convert seconds to milliseconds

    # Print the elapsed time in milliseconds
    print(f"Inference and communication took {elapsed_time:.2f} ms")
    # Assume MCU sends back 10 bytes for class probabilities
    pred = np.frombuffer(pred, dtype=np.uint8)
    print(f"MCU predictions:{pred}")
    mcu_predicted_class = np.argmax(pred)
    print(f'MCU Predicted Class: {classes[mcu_predicted_class]}')

    # Compare local and MCU predictions
    # if local_predicted_class == mcu_predicted_class:
    #     print("Success! Both predictions match.")
    # else:
    #     print("Mismatch in predictions. Local: {}, MCU: {}".format(classes[local_predicted_class], classes[mcu_predicted_class]))

    # Close the serial port
    ser.close()

if __name__ == '__main__':
    main()
