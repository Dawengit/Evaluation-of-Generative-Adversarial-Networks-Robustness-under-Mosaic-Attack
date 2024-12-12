from __future__ import print_function, division
from keras.layers import *
# from keras.layers.advanced_activations import LeakyReLU
# from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.layers import LeakyReLU, Conv2D, Add, BatchNormalization, Activation, Dense, Input
from keras.layers import UpSampling2D, Conv2D

# from tensorflow.keras.convolution import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Model
# from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam, legacy as optimizers_legacy
import datetime
import matplotlib.pyplot as plt
from data_loader import DataLoader
import numpy as np
import os

from PIL import Image
import numpy as np
import os
#----------------------------------------------------------

# Reference from https://github.com/TianLin0509/Easiest-SRGAN-demo
# chatgpt
# copilot
#----------------------------------------------------------

#----------------------------------------------------------

class SRGAN():
    def __init__(self):
        # Input shape
        self.channels = 3
        self.lr_height = 64  # Low resolution height
        self.lr_width = 64  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4  # High resolution height
        self.hr_width = self.lr_width * 4  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        # optimizer = Adam(0.0002, 0.5)
        optimizer_d = Adam(0.0002, 0.5)
        optimizer_g = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.summary()

        # Configure data loader
        # self.dataset_name = 'images_dataset_attacked'
        self.dataset_name = 'blurred_face_data_train_att'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

     
        # Calculate the output shape of the discriminator (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of generator and discriminator
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
                                   optimizer=optimizer_d,
                                   metrics=['accuracy'])
        self.discriminator.summary()

        # Build the generator
        self.generator = self.build_generator()

        # High resolution and low resolution image inputs
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high resolution image from low resolution image
        fake_hr = self.generator(img_lr)

        # Extract features of the generated image
        fake_features = self.vgg(fake_hr)

        # For the combined model, only train the generator
        self.discriminator.trainable = False

        # The discriminator determines the validity of the generated high resolution image
        validity = self.discriminator(fake_hr)

        # Define the combined model
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.summary()

        # Compile the combined model
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer_g)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model.
        """
        vgg = VGG19(weights="imagenet", include_top=False, input_shape=(256, 256, 3))
        # vgg.outputs = [vgg.get_layer("block3_conv4").output]
        # vgg.trainable = False

        img = Input(shape=self.hr_shape)
        vgg_layer_output = vgg.get_layer('block3_conv4').output  # Example layer name
        vgg_model = Model(inputs=vgg.input, outputs=vgg_layer_output)
        img_features = vgg_model(img)
        return Model(img, img_features)

    def build_generator(self):

        def residual_block(layer_input, filters):
            """Residual block described in paper"""
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propagate through residual blocks
        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input image
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)

        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)

        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)

        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)


    def train(self, epochs, batch_size=1, sample_interval=5):
        start_time = datetime.datetime.now()

        # Initialize lists to record losses
        d_losses = []
        g_losses = []

        for epoch in range(epochs):
            if epoch > 30:
                sample_interval = 10
            if epoch > 100:
                sample_interval = 50

            # ----------------------
            #  Train Discriminator
            # ----------------------

            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            fake_hr = self.generator.predict(imgs_lr)
            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            valid = np.ones((batch_size,) + self.disc_patch)
            image_features = self.vgg.predict(imgs_hr)
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            # Record the average loss for the current epoch
            d_losses.append(d_loss[0])
            # Here, g_loss[0] is the total loss
            g_losses.append(g_loss[0])

            elapsed_time = datetime.datetime.now() - start_time
            # Print the progress
            print(f"{epoch} time: {elapsed_time} [D loss: {d_loss[0]:.5f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss[0]:.5f}]")

            if epoch % sample_interval == 0:
                self.sample_images_new(epoch)
            if epoch % 100 == 0 and epoch > 1:
                self.generator.save_weights('./saved_model/' + str(epoch) + 'attacked' + '.h5')

            # Save the loss value into a file
            with open('losses.txt', 'w') as f:
                f.write(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss[0]}\n")

        # Plot loss values
        plt.figure(figsize=(10,5))
        plt.plot(d_losses, label="Discriminator Loss")
        plt.plot(g_losses, label="Generator Loss")
        plt.title("Training Loss", fontsize=34)
        plt.xlabel("Epoch", fontsize=32)
        plt.ylabel("Loss", fontsize=32)
        plt.legend()
        plt.savefig("loss_plot2.png")
        plt.close()


#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
    def test_images_2(self, batch_size=1):
        # Get HR and LR images to be predicted from the data loader (HR is only for comparison here)
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size, is_pred=True)
        
        # Create directory if it does not exist
        os.makedirs('saved_model/', exist_ok=True)

        # Load the trained generator weights
        # self.generator.load_weights('./saved_model/' + str(100)+str("attacked") + '.h5')

        self.generator.load_weights('./saved_model/' + str(200) + '.h5')
        
        # Use the generator to perform super-resolution on low resolution images
        fake_hr = self.generator.predict(imgs_lr)
        
        # Convert images from [-1,1] range back to [0,1]
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        
        # Save the generated fake images for each input image
        for i in range(fake_hr.shape[0]):
            # fake_hr[i] is a single image array with shape, for example, (height, width, channels)
            # Convert it to integer pixel values in the range 0-255
            img_array = (fake_hr[i] * 255).astype(np.uint8)
            # Save the image
            # You can change the file naming method as needed
            save_path = f"./fake_{i}.png"
            Image.fromarray(img_array).save(save_path)
            print(f"Fake image saved at: {save_path}")

        # Save low resolution images
        for i in range(imgs_lr.shape[0]):
            img_array = (imgs_lr[i] * 255).astype(np.uint8)
            save_path = f"./low_{i}.png"
            Image.fromarray(img_array).save(save_path)
            print(f"Low resolution image saved at: {save_path}")


#--------------------------------------------------------------------------


if __name__ == '__main__':
    gan = SRGAN()
    gan.train(epochs=205, batch_size=10, sample_interval=2)
    # gan.test_images_2() 