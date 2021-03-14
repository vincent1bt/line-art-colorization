import tensorflow as tf
from datetime import datetime
from glob import glob
import os

import argparse

from data_generators import InferenceDataGenerator

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
parser.add_argument('--save_images', default=True, action='store_true', help='Save generated images')

if not os.path.exists('test_shots/color'):
  raise("test_shots/color folder does not exists")

if not os.path.exists('test_shots/line_art_shots'):
  raise("test_shots/line_art_shots folder does not exists")

opts = parser.parse_args()

color_network = tf.keras.models.load_model('weights/saved_model/color')

folder_shots_names = os.listdir("test_shots/color")

batch_size = opts.batch_size

test_generator = InferenceDataGenerator(folder_shots_names, batch_size=batch_size)

generated_images_path = "generated_images"

if not os.path.exists(generated_images_path):
  os.makedirs(generated_images_path)

def save_images(y_trans):
  for index, img_trans in enumerate(y_trans):
    date_now = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    img_path = f"{generated_images_path}/Generated_Image_{index}_{date_now}.jpg"
    tf.keras.preprocessing.image.save_img(img_path, img_trans)    

for reference_0, middle, reference_1 in test_generator:
  y_trans, _, _ = color_network([middle[0], middle[1], 
                                 reference_0[0], reference_0[1], reference_0[2], 
                                 reference_1[0], reference_1[1], reference_1[2]])
  
  if opts.save_images:
    save_images(y_trans)
                                            