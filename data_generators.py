import os
import tensorflow as tf
import numpy as np

class InferenceDataGenerator(tf.keras.utils.Sequence):
  def __init__(self, folders_paths, batch_size=4):
    self.image_shape = (256, 455, 3)
    self.batch_size = batch_size
    self.folders_paths = folders_paths
    self.on_epoch_end()

  def __len__(self):
    return len(self.folders_paths) // self.batch_size

  def __getitem__(self, index):
    index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
    batch = [self.folders_paths[k] for k in index]

    X = self.__get_data(batch)
    return X

  def on_epoch_end(self):
    self.index = np.arange(len(self.folders_paths))
  
  def load_image(self, image_path, resize=False):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    if resize:
      img = tf.image.resize(img, [256, 455])

    img = img / 255.0
    
    return img

  def __get_data(self, batch):
    reference_color_images_0 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_line_art_images_0 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_distance_maps_0 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)

    reference_color_images_1 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_line_art_images_1 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_distance_maps_1 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)

    middle_line_art = np.empty((self.batch_size, *self.image_shape))
    middle_distance_map = np.empty((self.batch_size, *self.image_shape))
    
    for i, shot_path in enumerate(batch):
      selected_images = sorted(os.listdir(f"test_shots/color/{shot_path}"))[:3]
      
      color_images_paths = [f"test_shots/color/{shot_path}/{s_i}" for s_i in [selected_images[0], selected_images[-1]]]
      line_art_images_paths = [f"test_shots/line_art_shots/{shot_path}/{s_i}" for s_i in selected_images]
      distance_map_images_paths = [f"test_shots/distance_map_shots/{shot_path}/{s_i}" for s_i in selected_images]
      
      reference_color_images_0[i,] = self.load_image(color_images_paths[0], resize=True)
      reference_line_art_images_0[i,] = self.load_image(line_art_images_paths[0], resize=True)
      reference_distance_maps_0[i,] = self.load_image(distance_map_images_paths[0])

      middle_line_art[i,] = self.load_image(line_art_images_paths[1], resize=True)
      middle_distance_map[i,] = self.load_image(distance_map_images_paths[1])

      reference_color_images_1[i,] = self.load_image(color_images_paths[1], resize=True)
      reference_line_art_images_1[i,] = self.load_image(line_art_images_paths[2], resize=True)
      reference_distance_maps_1[i,] = self.load_image(distance_map_images_paths[2])

      concated_image = tf.concat([reference_color_images_0,
                                reference_line_art_images_0,
                                reference_distance_maps_0,
                                middle_line_art,
                                middle_distance_map,
                                reference_color_images_1,
                                reference_line_art_images_1,
                                reference_distance_maps_1], axis=0)

      cropped_image = tf.image.crop_to_bounding_box(concated_image, 0, 99, 256, 256)

    rf_0 = cropped_image[0:(self.batch_size)], cropped_image[(self.batch_size):(self.batch_size * 2)], cropped_image[(self.batch_size * 2):(self.batch_size * 3)]
    mid = cropped_image[(self.batch_size * 3):(self.batch_size * 4)], cropped_image[(self.batch_size * 4):(self.batch_size * 5)]
    rf_1 = cropped_image[(self.batch_size * 5):(self.batch_size * 6)], cropped_image[(self.batch_size * 6):(self.batch_size * 7)], cropped_image[(self.batch_size * 7):(self.batch_size * 8)]

    return rf_0, mid, rf_1

class DataGenerator(tf.keras.utils.Sequence):
  def __init__(self, folders_paths, batch_size=4, shuffle=True):
    self.image_shape = (256, 455, 3)
    self.batch_size = batch_size
    self.folders_paths = folders_paths
    self.shuffle = shuffle
    self.on_epoch_end()

  def __len__(self):
    return len(self.folders_paths) // self.batch_size

  def __getitem__(self, index):
    index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
    batch = [self.folders_paths[k] for k in index]

    X = self.__get_data(batch)
    return X

  def on_epoch_end(self):
    self.index = np.arange(len(self.folders_paths))
    if self.shuffle == True:
        np.random.shuffle(self.index)

  def get_random_positions(self, images_name):
    limit = len(images_name) - 4
    position = np.random.randint(0, limit)

    return [images_name[position], images_name[position + 2], images_name[position + 4]]
  
  def load_image(self, image_path, resize=False):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    if resize:
      img = tf.image.resize(img, [256, 455])

    img = img / 255.0
    
    return img

  def __get_data(self, batch):
    reference_color_images_0 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_line_art_images_0 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_distance_maps_0 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)

    reference_color_images_1 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_line_art_images_1 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)
    reference_distance_maps_1 = np.empty((self.batch_size, *self.image_shape), dtype=np.float32)

    middle_color = np.empty((self.batch_size, *self.image_shape))
    middle_line_art = np.empty((self.batch_size, *self.image_shape))
    middle_distance_map = np.empty((self.batch_size, *self.image_shape))
    
    for i, shot_path in enumerate(batch):
      images_name = sorted(os.listdir(f"final_shots/{shot_path}"))
      selected_images = self.get_random_positions(images_name)
      
      color_images_paths = [f"final_shots/{shot_path}/{s_i}" for s_i in selected_images]
      line_art_images_paths = [f"line_art_shots/{shot_path}/{s_i}" for s_i in selected_images]
      distance_map_images_paths = [f"distance_map_shots/{shot_path}/{s_i}" for s_i in selected_images]
      
      reference_color_images_0[i,] = self.load_image(color_images_paths[0], resize=True)
      reference_line_art_images_0[i,] = self.load_image(line_art_images_paths[0], resize=True)
      reference_distance_maps_0[i,] = self.load_image(distance_map_images_paths[0])

      middle_color[i,] = self.load_image(color_images_paths[1], resize=True)
      middle_line_art[i,] = self.load_image(line_art_images_paths[1], resize=True)
      middle_distance_map[i,] = self.load_image(distance_map_images_paths[1])

      reference_color_images_1[i,] = self.load_image(color_images_paths[2], resize=True)
      reference_line_art_images_1[i,] = self.load_image(line_art_images_paths[2], resize=True)
      reference_distance_maps_1[i,] = self.load_image(distance_map_images_paths[2])

      concated_image = tf.concat([reference_color_images_0,
                                reference_line_art_images_0,
                                reference_distance_maps_0,
                                middle_color,
                                middle_line_art,
                                middle_distance_map,
                                reference_color_images_1,
                                reference_line_art_images_1,
                                reference_distance_maps_1], axis=0)
    

      cropped_image = tf.image.crop_to_bounding_box(concated_image, 0, 99, 256, 256)

    
    rf_0 = cropped_image[0:4], cropped_image[4:8], cropped_image[8:12]
    mid = cropped_image[12:16], cropped_image[16:20], cropped_image[20:24]
    rf_1 = cropped_image[24:28], cropped_image[28:32], cropped_image[32:36]

    return rf_0, mid, rf_1