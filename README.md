# Deep Convolutional Generative Adversarial Network (DCGAN) for Handwritten Digits

## Project Overview

This repository implements a Deep Convolutional Generative Adversarial Network (DCGAN) in TensorFlow for generating handwritten digits, specifically using the MNIST dataset.

## Description

DCGAN is a type of Generative Adversarial Network (GAN) that uses convolutional neural networks to generate images. This project applies DCGAN to the MNIST dataset, allowing for the generation of realistic handwritten digits.

## Features

- Implementation of DCGAN architecture using TensorFlow
- Training on MNIST dataset for handwritten digit generation
- Customizable hyperparameters for experimentation
- Easy-to-use code for generating new digits

## Implementation Details

- Generator network: Convolutional transpose layers with batch normalization, Drop Out and Leaky ReLU activation
- Discriminator network: Convolutional layers with batch normalization, Drop Out and Leaky ReLU activation
- Loss functions: Binary cross-entropy loss
- Optimization: Adam optimizer with learning rate 2e-4
## Requirements

- TensorFlow 2.x
- Python 3.x
- NumPy
- Matplotlib (for visualization)

Usage

1. Clone repository
2. Install requirements
3. Download MNIST dataset
4. Run training script
5. Generate new digits using trained model

Example Use Cases

- Generating realistic handwritten digits for data augmentation
- Improving performance of handwritten digit recognition models
- Understanding GANs and their applications



