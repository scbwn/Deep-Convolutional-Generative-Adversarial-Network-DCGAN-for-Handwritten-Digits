import os
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers import Dense, Activation, Dropout
import numpy as np

class Training_Callback(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, saving_rate):
        super(Training_Callback, self).__init__()
        self.latent_dim = latent_dim
        self.saving_rate = saving_rate
        
    # Save Image sample from Generator
    def save_imgs(self, epoch):
        # Number of images = 16
        seed = tf.random.normal([16, self.latent_dim])
        gen_imgs = self.model.generator(seed, training=False)
        
        fig = plt.figure(figsize=(4, 4))

        for i in range(gen_imgs.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(gen_imgs[i, :, :, 0]*127.5+127.5, cmap='gray')
            plt.axis('off')

        #fig.savefig("./images/mnist_%d.png" % epoch)
    
    # Called after each epoch
    def on_epoch_end(self, epoch, logs=None):
        # Save image after 50 epochs
        if epoch % 2 == 0:
            self.save_imgs(epoch)
            
        if epoch > 0 and epoch % self.saving_rate == 0:
            save_dir = "./models/model_epoch_" + str(epoch)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.model.discriminator.save_weights(save_dir + '/discriminator_%d' % epoch)
            self.model.generator.save_weights(save_dir + '/generator_%d' % epoch)
            
        self.best_weights = self.model.get_weights()
        
        
        
class GAN(tf.keras.Model):
    # define the models
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
    
    # Define the compiler
    def compile(self, disc_optimizer, gen_optimizer, loss_fn, generator_loss, discriminator_loss):
        super(GAN, self).compile()
        self.disc_optimizer = disc_optimizer
        self.gen_optimizer = gen_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.loss_fn = loss_fn

        
    # @tf.function: The below function is completely Tensor Code
    # Good for optimization
    @tf.function
    # Modify Train step for GAN
    def train_step(self, images):
        batch_size = tf.shape(images)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        # Define the loss function
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = gen(noise, training=True)
            real_output = disc(images, training=True)
            fake_output = disc(generated_images, training=True)
            gen_loss = generator_loss(self.loss_fn, fake_output)
            disc_loss = discriminator_loss(self.loss_fn, real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(gen_loss, gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, disc.trainable_variables)
        gen_optimizer.apply_gradients(zip(gradients_of_generator, gen.trainable_variables))
        disc_optimizer.apply_gradients(zip(gradients_of_discriminator, disc.trainable_variables))
        
        return {"Gen Loss ": gen_loss,"Disc Loss" : disc_loss}