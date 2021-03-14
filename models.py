import tensorflow as tf
from tensorflow.keras import layers

from model_blocks import SEV, Decoder, ResBlock, ColorEncoder, LineArtEncoder, DistanceMapEncoder, ColorTransformLayer, SpectralNormalization

class ColorTransformNetwork(tf.keras.Model):
  def __init__(self):
    super(ColorTransformNetwork, self).__init__()
  
    self.color_encoder = ColorEncoder()
    self.lineart_encoder = LineArtEncoder()
    self.distance_encoder = DistanceMapEncoder()

    self.color_transform_layer = ColorTransformLayer()
    self.sev = SEV()

    self.res_block_1 = ResBlock()
    self.res_block_2 = ResBlock()
    self.res_block_3 = ResBlock()
    self.res_block_4 = ResBlock()
    self.res_block_5 = ResBlock()
    self.res_block_6 = ResBlock()
    self.res_block_7 = ResBlock()
    self.res_block_8 = ResBlock()

    self.sim_conv = layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same')
    self.mid_conv = layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same')

    self.decoder = Decoder()
  
  def call(self, inputs):
    target_line_art_images = inputs[0]
    target_distance_maps = inputs[1]
    reference_color_images_0 = inputs[2]
    reference_line_art_images_0 = inputs[3]
    reference_distance_maps_0 = inputs[4]
    reference_color_images_1 = inputs[5]
    reference_line_art_images_1 = inputs[6]
    reference_distance_maps_1 = inputs[7]

    target_line_art_images_features = self.lineart_encoder(target_line_art_images) # EnL
    target_distance_maps_features = self.distance_encoder(target_distance_maps) # EnD

    reference_distance_maps_0_features = self.distance_encoder(reference_distance_maps_0) # EnD
    reference_distance_maps_1_features = self.distance_encoder(reference_distance_maps_1) # EnD

    reference_color_images_0_features = self.color_encoder(reference_color_images_0) # EnC
    reference_color_images_1_features = self.color_encoder(reference_color_images_1) # EnC

    f_sim = self.color_transform_layer([target_distance_maps_features,
                                  reference_distance_maps_0_features,
                                  reference_distance_maps_1_features,
                                  reference_color_images_0_features,
                                  reference_color_images_1_features])

    style_vector = self.sev([reference_line_art_images_0,
                            reference_color_images_0,
                            reference_line_art_images_1,
                            reference_color_images_1]) # [Batch, 1, 1, 512])

    Y_trans_sim = self.sim_conv(f_sim) # [Batch, 64, 64, 3]
    Y_trans_sim = layers.UpSampling2D(size=(2, 2))(Y_trans_sim)
    Y_trans_sim = layers.UpSampling2D(size=(2, 2))(Y_trans_sim) # [Batch, 256, 256, 3]
    
    res_input = layers.add([target_line_art_images_features, f_sim]) # [Batch, 64, 64, 256]

    x = self.res_block_1([res_input, style_vector])
    x = self.res_block_2([x, style_vector])
    x = self.res_block_3([x, style_vector])
    x = self.res_block_4([x, style_vector])
    x = self.res_block_5([x, style_vector])
    x = self.res_block_6([x, style_vector])
    x = self.res_block_7([x, style_vector])
    x = self.res_block_8([x, style_vector])

    Y_trans_mid = self.mid_conv(x)
    Y_trans_mid = layers.UpSampling2D(size=(2, 2))(Y_trans_mid)
    Y_trans_mid = layers.UpSampling2D(size=(2, 2))(Y_trans_mid)

    Y_trans = self.decoder(x)

    return Y_trans, Y_trans_mid, Y_trans_sim

  
class ColorTransformDiscriminator(tf.keras.Model):
  def __init__(self):
    super(ColorTransformDiscriminator, self).__init__()
    self.conv_1 = SpectralNormalization(layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    self.conv_2 = SpectralNormalization(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    self.conv_3 = SpectralNormalization(layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"))
    self.conv_4 = SpectralNormalization(layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"))
    self.conv_5 = SpectralNormalization(layers.Conv2D(1, (3, 3), strides=(2, 2), padding="same"))

    self.activation = layers.LeakyReLU(0.2)

  def call(self, line_art, target_y_trans):
    x = tf.concat([line_art, target_y_trans], -1) # Bx256x256x6
    x = self.conv_1(x) # 128
    x = self.activation(x)
    x = self.conv_2(x) # 64
    x = self.activation(x)
    x = self.conv_3(x) # 32
    x = self.activation(x)
    x = self.conv_4(x) # 16
    x = self.activation(x)
    x = self.conv_5(x) # 8

    return x
  
class Vgg19(tf.keras.Model):
  def __init__(self):
    super(Vgg19, self).__init__()
    layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] 
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layers]

    self.model = tf.keras.Model([vgg.input], outputs)
  
  def call(self, x):
    x = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
    return self.model(x)