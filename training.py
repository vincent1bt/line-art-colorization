import os
import time
import argparse

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train the model')
parser.add_argument('--continue_training', default=False, action='store_true', help='Use training checkpoints to continue training the model')

opts = parser.parse_args()

from data_generators import DataGenerator
from models import ColorTransformNetwork, ColorTransformDiscriminator, Vgg19
from loss_functions import compute_color_network_loss, compute_discriminator_2d_loss

name_folders = os.listdir("line_art_shots")

train_generator = DataGenerator(name_folders)

generator_lr = 1e-4
discriminator_lr = 1e-5

color_network_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr, beta_1=0.5, beta_2=0.999)
discriminator_2d_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr, beta_1=0.5, beta_2=0.999)

batch_size = 4
train_steps = len(name_folders) // batch_size

color_network = ColorTransformNetwork()
discriminator_2d = ColorTransformDiscriminator()
vgg = Vgg19()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(color_network=color_network,
                                 discriminator_2d=discriminator_2d,
                                 color_network_optimizer=color_network_optimizer,
                                 discriminator_2d_optimizer=discriminator_2d_optimizer)


if opts.continue_training:
  print("loading training checkpoints: ")                   
  print(tf.train.latest_checkpoint(checkpoint_dir))
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

@tf.function
def train_step(x, y0, y1):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    y_trans, y_trans_mid, y_trans_sim = color_network([x[1], x[2], y0[0], y0[1], y0[2], y1[0], y1[1], y1[2]], training=True)

    y = x[0]

    y_list = vgg(y)
    y_trans_list = vgg(y_trans)

    y_class = discriminator_2d(x[1], y, training=True)
    y_trans_class = discriminator_2d(x[1], y_trans, training=True)

    color_network_loss, gen_loss, latent_loss, s_loss, p_loss, l_loss = compute_color_network_loss(y_trans_class, y, y_trans, y_trans_sim, y_trans_mid, y_list, y_trans_list)
    discriminator_2d_loss = compute_discriminator_2d_loss(y_class, y_trans_class)
  
  color_network_gradients = gen_tape.gradient(color_network_loss, color_network.trainable_variables)
  discriminator_2d_gradients = disc_tape.gradient(discriminator_2d_loss, discriminator_2d.trainable_variables)

  color_network_optimizer.apply_gradients(zip(color_network_gradients, color_network.trainable_variables))
  discriminator_2d_optimizer.apply_gradients(zip(discriminator_2d_gradients, discriminator_2d.trainable_variables))

  return color_network_loss, gen_loss, latent_loss, s_loss, p_loss, l_loss, discriminator_2d_loss

train_loss_results = []
generator_loss_results = []
discriminator_loss_results = []

gen_loss_results = []
latent_loss_results = []
s_loss_results = []
p_loss_results = []
l_loss_results = []

def plot_metrics(train_loss, generator_loss, discriminator_loss, gen_loss, latent_loss, s_loss, p_loss, l_loss):
  fig, ax = plt.subplots(2, 4, figsize=(20, 20))

  ax[0, 0].plot(np.arange(len(train_loss)), train_loss)
  ax[0, 0].set_title('train_loss')

  ax[0, 1].plot(np.arange(len(generator_loss)), generator_loss)
  ax[0, 1].set_title('generator_loss')

  ax[0, 2].plot(np.arange(len(discriminator_loss)), discriminator_loss)
  ax[0, 2].set_title('discriminator_loss')

  ax[0, 3].plot(np.arange(len(gen_loss)), gen_loss)
  ax[0, 3].set_title('gen_loss')

  ax[1, 0].plot(np.arange(len(latent_loss)), latent_loss)
  ax[1, 0].set_title('latent_loss')

  ax[1, 1].plot(np.arange(len(s_loss)), s_loss)
  ax[1, 1].set_title('s_loss')

  ax[1, 2].plot(np.arange(len(p_loss)), p_loss)
  ax[1, 2].set_title('p_loss')

  ax[1, 3].plot(np.arange(len(l_loss)), l_loss)
  ax[1, 3].set_title('l_loss')

def train():
  for epoch in range(epochs):
    batch_time = time.time()
    epoch_time = time.time()
    step = 0
    epoch_count = f"0{epoch + 1}/{epochs}" if epoch < 9 else f"{epoch + 1}/{epochs}"

    for reference_0, middle, reference_1 in train_generator:
      color_network_loss, gen_loss, latent_loss, s_loss, p_loss, l_loss, discriminator_2d_loss = train_step(middle, reference_0, reference_1)
      
      color_network_loss = float(color_network_loss)
      discriminator_2d_loss = float(discriminator_2d_loss)
      loss = color_network_loss + discriminator_2d_loss
      step += 1

      print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
              '| Loss:', f"{loss:.5f}", '| Discriminator loss:', f"{discriminator_2d_loss:.5f}",
             '| Generator loss:', f"{color_network_loss:.5f}", "| Step Time:", f"{time.time() - batch_time:.2f}", end='')    
        
      batch_time = time.time()
      train_loss_results.append(loss)
      generator_loss_results.append(color_network_loss)
      discriminator_loss_results.append(discriminator_2d_loss)
      gen_loss_results.append(float(gen_loss))
      latent_loss_results.append(float(latent_loss))
      s_loss_results.append(float(s_loss))
      p_loss_results.append(float(p_loss))
      l_loss_results.append(float(l_loss))
      
    checkpoint.save(file_prefix=checkpoint_prefix)

    print('\r', 'Epoch', epoch_count, '| Step', f"{step}/{train_steps}",
          '| Loss:', f"{loss:.5f}", '| Discriminator loss:', f"{discriminator_2d_loss:.5f}",
          '| Generator loss:', f"{color_network_loss:.5f}", "| Epoch Time:", f"{time.time() - epoch_time:.2f}")

epochs = opts.epochs
print(f"Training for {epochs} epochs")
train()

color_network.save("weights/saved_model/color", include_optimizer=False)