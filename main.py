import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Activation, Dropout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import time

from dcgan import Training_Callback, GAN

# Function definitions
def create_generator():    
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 512)))
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model

def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

def discriminator_loss(loss_object, real_output, fake_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    fake_loss = loss_object(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(loss_object, discriminator_probability):
    return loss_object(tf.ones_like(discriminator_probability), discriminator_probability)


# Hyperparameters
latent_dim = 100
epochs = 10
batch_size = 256

# Save model after every *saving_rate* epochs
saving_rate = 100

# Random Seed for Shuffling Data
buffer_size = 5000

# Optimizers for generator and discriminator
gen_optimizer = tf.keras.optimizers.Adam(0.0002)
disc_optimizer = tf.keras.optimizers.Adam(0.0002)

# Load data
from tf.keras.datasets import mnist
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5

# Shuffle & Batch Data
train_dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(buffer_size , reshuffle_each_iteration=True).batch(batch_size)

# Define Loss Function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Create generator and discriminator
disc = create_discriminator()
gen = create_generator()

gan = GAN(discriminator=disc, generator=gen, latent_dim=latent_dim)
gan.compile(
    disc_optimizer=disc_optimizer,
    gen_optimizer=gen_optimizer,
    loss_fn=cross_entropy,
    generator_loss = generator_loss,
    discriminator_loss = discriminator_loss
)

# Callback to save images during training
training_callback = Training_Callback(latent_dim, saving_rate)

# Train GAN
gan.fit(
    train_dataset, 
    epochs=epochs,
    callbacks=[training_callback]
)