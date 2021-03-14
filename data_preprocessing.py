import os
from glob import glob

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import cv2
from scipy import ndimage

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--generate_test_files', default=False, action='store_true', help='Generates The line art and distance map images to test the network')
parser.add_argument('--generate_distance_map_only', default=False, action='store_true', help='Generates Only the distance map images from the test_shots folder')

opts = parser.parse_args()

if opts.generate_test_files and not os.path.exists('test_shots/color'):
  raise("test_shots/color folder does not exists")

if not opts.generate_test_files and not os.path.exists('final_shots'):
  raise("final_shots folder does not exists")

## Sketch Keras
def create_sketch_model(input_shape=(1024, 1024, 1)):
  input = layers.Input(shape=input_shape)
  model = layers.Conv2D(32, (3, 3), padding='same')(input)
  model = layers.BatchNormalization()(model)
  activation_1 = layers.ReLU()(model)

  model = layers.Conv2D(64, (4, 4), (2, 2), padding='same')(activation_1)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(64, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_3 = layers.ReLU()(model)

  model = layers.Conv2D(128, (4, 4), (2, 2), padding='same')(activation_3)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(128, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_5 = layers.ReLU()(model)

  model = layers.Conv2D(256, (4, 4), (2, 2), padding='same')(activation_5)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(256, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_7 = layers.ReLU()(model)

  model = layers.Conv2D(512, (4, 4), (2, 2), padding='same')(activation_7)
  model = layers.BatchNormalization()(model)
  activation_8 = layers.ReLU()(model)

  model = layers.Conv2D(512, (3, 3), padding='same')(activation_8)
  model = layers.BatchNormalization()(model)
  activation_9 = layers.ReLU()(model)

  model = layers.concatenate([activation_8, activation_9])

  model = layers.UpSampling2D()(model)

  model = layers.Conv2D(512, (4, 4), padding='same')(model)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(256, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_11 = layers.ReLU()(model)

  model = layers.concatenate([activation_7, activation_11])

  model = layers.UpSampling2D()(model)

  model = layers.Conv2D(256, (4, 4), padding='same')(model)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(128, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_13 = layers.ReLU()(model)

  model = layers.concatenate([activation_5, activation_13])

  model = layers.UpSampling2D()(model)

  model = layers.Conv2D(128, (4, 4), padding='same')(model)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(64, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_15 = layers.ReLU()(model)

  model = layers.concatenate([activation_3, activation_15])

  model = layers.UpSampling2D()(model)

  model = layers.Conv2D(64, (4, 4), padding='same')(model)
  model = layers.BatchNormalization()(model)
  model = layers.ReLU()(model)

  model = layers.Conv2D(32, (3, 3), padding='same')(model)
  model = layers.BatchNormalization()(model)
  activation_17 = layers.ReLU()(model)

  model = layers.concatenate([activation_1, activation_17])

  model = layers.Conv2D(1, (3, 3), padding='same')(model)

  return tf.keras.Model(inputs=input, outputs=[model])

if not opts.generate_distance_map_only:
  sketchKeras = create_sketch_model()
  sketchKeras.load_weights("weights/sketchKeras.h5")

def get_light_map_single(img): # Used
  gray = img
  gray = gray[None]
  gray = gray.transpose((1,2,0))
  blur = cv2.GaussianBlur(gray, (0, 0), 3)
  gray = gray.reshape((gray.shape[0],gray.shape[1]))
  highPass = gray.astype(int) - blur.astype(int)
  highPass = highPass.astype(np.float)
  highPass = highPass / 128.0
  return highPass

def normalize_pic(img): # used
  img = img / np.max(img)
  return img

def resize_img_512_3d(img): # used
    zeros = np.zeros((1,3,1024,1024), dtype=np.float)
    zeros[0 , 0 : img.shape[0] , 0 : img.shape[1] , 0 : img.shape[2]] = img
    return zeros.transpose((1,2,3,0)) # BxCxHxW to CxHxWxB

def show_active_img_and_save_denoise(img, dir_path):
    mat = img.astype(np.float)
    mat = - mat + 1
    mat = mat * 255.0
    mat[mat < 0] = 0
    mat[mat > 255] = 255
    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat, 1)
    # mat = cv2.resize(mat, (455, 256))
    mat = cv2.resize(mat, (1024, 576))
    cv2.imwrite(dir_path, mat)
    return mat

def get_image(img_path):
  from_mat = cv2.imread(img_path)
  width = float(from_mat.shape[1])
  height = float(from_mat.shape[0])
  new_width = 0
  new_height = 0

  if (width > height):
      from_mat = cv2.resize(from_mat, (1024, int(1024 / width * height)), interpolation=cv2.INTER_AREA)
      new_width = 1024
      new_height = int(1024 / width * height)
  else:
      from_mat = cv2.resize(from_mat, (int(1024 / height * width), 1024), interpolation=cv2.INTER_AREA)
      new_width = int(1024 / height * width)
      new_height = 1024

  from_mat = from_mat.transpose((2, 0, 1)) # HxWxC to CxHxW
  light_map = np.zeros(from_mat.shape, dtype=np.float)

  for channel in range(3):
    light_map[channel] = get_light_map_single(from_mat[channel])

  light_map = normalize_pic(light_map)
  light_map = resize_img_512_3d(light_map)

  return light_map, new_width, new_height

def get_line_art(line_mat, dir_path, new_width, new_height):
  line_mat = line_mat.transpose((3, 1, 2, 0))[0] # CxHxWxB to BxHxWxC
  line_mat = line_mat[0:int(new_height), 0:int(new_width), :]

  image = np.amax(line_mat, 2)

  show_active_img_and_save_denoise(image, dir_path)

line_art_folder_path = "line_art_shots"
folders_path = "final_shots"
distance_map_folder_path = "distance_map_shots"

if opts.generate_test_files:
  line_art_folder_path = "test_shots/line_art_shots"
  folders_path = "test_shots/color"
  distance_map_folder_path = "test_shots/distance_map_shots"

if not os.path.exists(line_art_folder_path):
  if not opts.generate_distance_map_only:
    os.makedirs(line_art_folder_path)
  else:
    raise("test_shots/line_art_shots does not exists")

folders = glob(f"{folders_path}/*")

## Create line art images
print("Creating Line Art Images")

if not opts.generate_distance_map_only:
  for folder in folders:
    name = folder.split("/")[-1]
    new_dir = f"{line_art_folder_path}/{name}"
    os.makedirs(new_dir)
    images = glob(f"./{folders_path}/{name}/*")

    img_stack = np.empty((len(images) * 3, 1024, 1024, 1))

    for idx, image in enumerate(images):
      img, new_width, new_height = get_image(image)
      img_stack[idx * 3:(idx + 1) * 3,] = img

    for index, image in enumerate(images):
      line_mat_stack = np.empty((3, 1024, 1024, 1))

      line_mat_stack = sketchKeras.predict(img_stack[index * 3:(index + 1) * 3,], verbose=0)
      image_name = image.split("/")[-1]
      dir_path = f"{new_dir}/{image_name}"

      get_line_art(line_mat_stack, dir_path, new_width, new_height)

if not os.path.exists(distance_map_folder_path):
  os.makedirs(distance_map_folder_path)

print("Creating Distance Map Images")

line_art_folders = glob(f"{line_art_folder_path}/*")

def binarize(sketch, threshold=127):
    return tf.where(sketch < threshold, x=tf.zeros_like(sketch), y=tf.ones_like(sketch) * 255.)

def get_distance_map(image_path, dir):
  img = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale")
  img = tf.keras.preprocessing.image.img_to_array(img)

  sketch = binarize(img)
  a = tf.cast(sketch, tf.uint8).numpy()
  a = a[:, :, 0]

  distance = ndimage.distance_transform_edt(a)
  distance = distance / tf.reduce_max(distance)

  final = (distance + (img[:,:,0] / 255.0) / 12)

  final = tf.image.resize(tf.expand_dims(final, axis=-1), [256, 455])

  tf.keras.preprocessing.image.save_img(dir, final)  

for folder in line_art_folders:
  name = folder.split("/")[-1]
  new_dir = f"{distance_map_folder_path}/{name}"
  os.makedirs(new_dir)
  images = glob(f"./{line_art_folder_path}/{name}/*")

  print('\r', len(images), end='')

  for image in images:
    image_name = image.split("/")[-1]
    dir_path = f"{new_dir}/{image_name}"

    get_distance_map(image, dir_path)
  
print("Data Ready")