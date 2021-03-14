import tensorflow as tf
from tensorflow.keras import layers

def l1_loss(y, y_trans):
  return tf.reduce_mean(tf.abs(y - y_trans))

def perceptual_loss(y_list, y_trans_list):
  loss = 0
  for feature_map_y, feature_map_y_trans in zip(y_list, y_trans_list):
    loss += tf.reduce_mean(tf.math.abs(feature_map_y - feature_map_y_trans))
  
  return (loss / 5) * 3e-2

def get_gram_matrix(feature_map):
  B, H, W, C = feature_map.shape
  matrix = tf.transpose(feature_map, [0, 3, 1, 2])
  matrix = tf.reshape(matrix, [B, C, H * W])

  num_locations = tf.cast(H * W, tf.float32)

  gram_matrix = tf.linalg.matmul(matrix, matrix, transpose_b=True) # C, HW * HW, C
  gram_matrix = gram_matrix / num_locations

  return gram_matrix


def style_loss(y_list, y_trans_list):
  loss = 0
  for feature_map_y, feature_map_y_trans in zip(y_list, y_trans_list):
    loss += tf.reduce_mean(tf.abs(get_gram_matrix(feature_map_y) - get_gram_matrix(feature_map_y_trans)))
  
  return (loss / 5) * 1e-6

def latent_constraint_loss(y, y_trans_sim, y_trans_mid):
  loss = tf.reduce_mean(tf.abs(y - y_trans_sim) + tf.abs(y - y_trans_mid))
  return loss

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(y_trans_class):
  return cross_entropy(tf.ones_like(y_trans_class), y_trans_class)


def compute_color_network_loss(y_trans_class, y, y_trans, y_trans_sim, y_trans_mid, y_list, y_trans_list, lambda_style=1000, lambda_l1=10):
  loss = 0
  gen_loss = generator_loss(y_trans_class)

  loss += gen_loss
  latent_loss = latent_constraint_loss(y, y_trans_sim, y_trans_mid)

  loss += latent_loss
  s_loss = style_loss(y_list, y_trans_list) * lambda_style

  loss += s_loss
  p_loss = perceptual_loss(y_list, y_trans_list)

  loss += p_loss
  l_loss = l1_loss(y, y_trans) * lambda_l1

  loss += l_loss

  return loss, gen_loss, latent_loss, s_loss, p_loss, l_loss

def compute_discriminator_2d_loss(y_class, y_trans_class):
  real_loss = cross_entropy(tf.ones_like(y_class), y_class)
  fake_loss = cross_entropy(tf.zeros_like(y_trans_class), y_trans_class)
  loss = real_loss + fake_loss

  return loss