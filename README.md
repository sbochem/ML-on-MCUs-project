# Image classifcation: Dataset augmentation and Latency-Accuracy Tradeoff

## Description
This project trains and quantizes different models on the CIFAR-10 and MNIST datasets. It also allows you to generate a dataset using a camera.

The folders `CIFAR_augmented_data` and `MNIST_augmented_data` contain the self-collected dataset augmentations for CIFAR and MNIST. These datasets contain 10 images for each of the 10 available classes. To load them into a notebook, use the `load_and_augment_data` function that can be found in the `cifar_train.ipynb` or the `load_data` function in `mnist_train.ipynb`.

The folder `CIFAR_inference` and `MNIST_inference` contain the STM32 project used for real-time image classification. These can be programmed onto the STM32 as usual.

`models` contains tflite models that were generated with `cifar_train.ipynb` or `mnist_train.ipynb`. These include FP32, INT8 and INT8 models found by Quantization-Aware Training (QAT).

The folder `ONNX_pretrained` contains pretrained LeNet and ResNet models pre-trained on CIFAR-10.

The scripts `inference_cifar_camera.py` and `inference_mnist_camera.py` can be used to perform real-time inference. For example, build and flash the  `CIFAR_inference` project on the STM32, then run inference_cifar_camera.py. Hold your image to be classified near the camera you plan to use. Right now, the build-in laptop camera should be used. The settings for this can be changed, here 

```python
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

ret, frame = cap.read()
cap.release()
if not ret:
    raise IOError("Cannot capture image from webcam")
```


The notebook `pruning_models.ipynb` contains several implementations of Pruning mostly from this source (https://arxiv.org/abs/2102.00554). These methods include Pruning based on weight magnitude (`prune_weights_magnitude`), Pruning based on the Loss function gradient with respect to the given weight (`prune_model_based_on_gradients`), pruning during training `train_model_pruning` and more. Unfortunately, these methods never really found their use as Pruning didn't help in reducing the memory footprint of the models.
