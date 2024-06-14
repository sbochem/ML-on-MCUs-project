# Image classifcation: Dataset augmentation and Latency-Accuracy Tradeoff

## Description
This project trains and quantizes different models on the CIFAR-10 and MNIST datasets. It also allows you to generate a dataset using a camera.

The folders `CIFAR_augmented_data` and `MNIST_augmented_data` contain the self-collected dataset augmentations for CIFAR and MNIST.

The folder `CIFAR_inference` and `MNIST_inference` contain the STM32 project used for real-time image classification.

`models` contains tflite models that were generated with `cifar_train.ipynb` or `mnist_train.ipynb`. 

The folder `ONNX_pretrained` contains pretrained LeNet and ResNet models pre-trained on CIFAR-10.

The scripts `inference_cifar_camera.py` and `inference_mnist_camera.py` can be used to perform real-time inference. For example, build and flash the CIFAR_inference project, then run inference_cifar_camera.py. Hold your image to be classified near the camera you plan to use. Right now, the build-in laptop camera should be used. 

The notebook `pruning_models.ipynb` contains several implementations of Pruning mostly from this source (https://arxiv.org/abs/2102.00554). Unfortunately, these methods never really found their use as Pruning didn't help in reducing the memory footprint of the models.
