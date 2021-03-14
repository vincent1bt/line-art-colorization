import tensorflow as tf
from tensorflow.keras import layers
from model_blocks import SpectralNormalization

class LearnableTSM(tf.keras.Model):
  def __init__(self):
    super(LearnableTSM, self).__init__()
    self.shift_ratio = 0.5
    self.shift_groups = 2
    self.shift_width = 3

    pre_weights = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)
    pre_weights = tf.reshape(pre_weights, [3, 1, 1, 1, 1])

    post_weights = tf.constant([1.0, 0.0, 0.0], dtype=tf.float32)
    post_weights = tf.reshape(post_weights, [3, 1, 1, 1, 1])

    self.pre_shift_conv = layers.Conv3D(1, [3, 1, 1], use_bias=False, padding="same", weights=[pre_weights])
    self.post_shift_conv = layers.Conv3D(1, [3, 1, 1], use_bias=False, padding="same", weights=[post_weights])

  def apply_tsm(self, tensor, conv):
    B, T, H, W, C = tensor.shape

    tensor = tf.transpose(tensor, [0, 4, 1, 2, 3])
    tensor = conv(tf.reshape(tensor, [B * C, T, H, W, 1]))
    tensor = tf.reshape(tensor, [B, C, T, H, W])
    tensor = tf.transpose(tensor, [0, 2, 3, 4, 1])

    return tensor

  def call(self, input_tensor):
    shape = B, T, H, W, C = input_tensor.shape
    split_size = int(C * self.shift_ratio) // self.shift_groups

    split_sizes = [split_size] * self.shift_groups + [C - split_size * self.shift_groups]
    tensors = tf.split(input_tensor, split_sizes, -1)
    assert len(tensors) == self.shift_groups + 1

    # we pass all the images here(full batch) but each image only contains a part of its channels
    tensor_1 = self.apply_tsm(tensors[0], self.pre_shift_conv)
    tensor_2 = self.apply_tsm(tensors[1], self.post_shift_conv)

    final_tensor = tf.concat([tensor_1, tensor_2, tensors[2]], -1)
    final_tensor = tf.reshape(final_tensor, shape)
    
    return final_tensor

class GatedConv(tf.keras.Model):
  def __init__(self, kernels, kernel_size, strides, dilation=(1, 1)):
    super(GatedConv, self).__init__()

    self.learnableTSM = LearnableTSM()
    self.feature_conv = SpectralNormalization(layers.Conv2D(kernels, kernel_size, strides=strides, padding="same", dilation_rate=dilation))

    self.gate_conv = SpectralNormalization(layers.Conv2D(kernels, kernel_size, strides=strides, padding="same", dilation_rate=dilation))

    self.activation = layers.LeakyReLU(0.2)
  
  def call(self, input_tensor):
    B, T, H, W, C = input_tensor.shape
    xs = tf.split(input_tensor, num_or_size_splits=T, axis=1)
    gating = tf.stack([self.gate_conv(tf.squeeze(x, axis=1)) for x in xs], axis=1)
    gating = tf.keras.activations.sigmoid(gating)

    feature = self.learnableTSM(input_tensor)
    # shape B, T, H, W, C

    feature = self.feature_conv(tf.reshape(feature, [B * T, H, W, C]))
    _, H_, W_, C_ = feature.shape
    feature = tf.reshape(feature, [B, T, H_, W_, C_])
    feature = self.activation(feature)

    out = gating * feature

    return out


class GatedDeConv(tf.keras.Model):
  def __init__(self, kernels):
    super(GatedDeConv, self).__init__()
    self.gate_conv = GatedConv(kernels, (3, 3), (1, 1))
    self.upsampling = layers.UpSampling3D(size=(1, 2, 2))
  
  def call(self, input_tensor):
    x = self.upsampling(input_tensor)
    x = self.gate_conv(x)

    return x

class TemporalConstraintNetwork(tf.keras.Model):
  def __init__(self):
    super(TemporalConstraintNetwork, self).__init__()
    self.conv_1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")
    self.conv_2 = GatedConv(64, (3, 3), (1, 1))
    self.conv_3 = GatedConv(128, (3, 3), (2, 2))
    self.conv_4 = GatedConv(256, (3, 3), (2, 2))

    self.dilation_1 = GatedConv(256, (3, 3), (1, 1), (2, 2)) # 2
    self.dilation_2 = GatedConv(256, (3, 3), (1, 1), (2, 2)) # 4
    self.dilation_3 = GatedConv(256, (3, 3), (1, 1), (2, 2)) # 8
    self.dilation_4 = GatedConv(256, (3, 3), (1, 1), (2, 2)) # 16

    self.conv_5 = GatedConv(256, (3, 3), (1, 1))
    self.up_conv_1 = GatedDeConv(128)
    self.up_conv_2 = GatedDeConv(3)
  
  def call(self, input_tensor):
    x = self.conv_1(input_tensor)
    x = self.conv_2(x) # Bx3x256x256x64
    x_1 = self.conv_3(x) # Bx3x128x128x128
    x_2 = self.conv_4(x_1) # Bx3x64x64x256

    x = self.dilation_1(x_2)
    x = self.dilation_2(x)
    x = self.dilation_3(x)
    x = self.dilation_4(x) # Bx3x64x64x256

    x = self.conv_5(x) # Bx3x64x64x256
    x = layers.concatenate([x, x_2], axis=-1) # or axis 1??
    x = self.up_conv_1(x)
    x = layers.concatenate([x, x_1], axis=-1) # or axis 1??
    x = self.up_conv_2(x)

    return x