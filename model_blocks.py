import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

class SpectralNormalization(layers.Wrapper):
  def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
    super(SpectralNormalization, self).__init__(layer, **kwargs)
    self.iteration = iteration
    self.eps = eps
    self.do_power_iteration = training
  
  def build(self, input_shape):
    self.layer.build(input_shape)
    self.w = self.layer.kernel
    self.w_shape = self.w.shape.as_list()

    self.v = self.add_weight(shape=(1, tf.math.reduce_prod(self.w_shape[:-1])),
                                    initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                    trainable=False,
                                    name='sn_v',
                                    dtype=tf.float32)
    
    self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                              initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                              trainable=False,
                              name='sn_u',
                              dtype=tf.float32)
  
    super(SpectralNormalization, self).build()
  
  def call(self, layer_inputs):
    self.update_weights()
    output = self.layer(layer_inputs)
    # self.restore_weights()

    return output

  def update_weights(self):
    w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
    
    u_hat = self.u
    v_hat = self.v

    if self.do_power_iteration:
      for _ in range(self.iteration):
        v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
        v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

        u_ = tf.matmul(v_hat, w_reshaped)
        u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)


    sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
    self.u.assign(u_hat)
    self.v.assign(v_hat)

    self.layer.kernel.assign(self.w / sigma)
      
  def restore_weights(self):
    self.layer.kernel.assign(self.w)

class ColorEncoder(tf.keras.Model):
  def __init__(self):
    super(ColorEncoder, self).__init__()
    self.conv_1 = SpectralNormalization(layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    self.conv_2 = SpectralNormalization(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    self.conv_3 = SpectralNormalization(layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"))

    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.activation = layers.ReLU()
  
  def call(self, x):
    x = self.conv_1(x) # 256
    x = self.norm_1(x)
    x = self.activation(x)
    x = self.conv_2(x) # 128
    x = self.norm_2(x)
    x = self.activation(x)
    x = self.conv_3(x) # 64
    x = self.norm_3(x)
    x = self.activation(x)

    return x # output Bx64x64x256

class LineArtEncoder(tf.keras.Model):
  def __init__(self):
    super(LineArtEncoder, self).__init__()
    self.conv_1 = SpectralNormalization(layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    self.conv_2 = SpectralNormalization(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    self.conv_3 = SpectralNormalization(layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"))

    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.activation = layers.ReLU()
  
  def call(self, x):
    x = self.conv_1(x) # 256
    x = self.norm_1(x)
    x = self.activation(x)
    x = self.conv_2(x) # 128
    x = self.norm_2(x)
    x = self.activation(x)
    x = self.conv_3(x) # 64
    x = self.norm_3(x)
    x = self.activation(x)

    return x # output Bx64x64x256

class DistanceMapEncoder(tf.keras.Model):
  def __init__(self):
    super(DistanceMapEncoder, self).__init__()
    self.conv_1 = SpectralNormalization(layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    self.conv_2 = SpectralNormalization(layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
    self.conv_3 = SpectralNormalization(layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"))

    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.activation = layers.ReLU()
  
  def call(self, x):
    x = self.conv_1(x) # 256
    x = self.norm_1(x)
    x = self.activation(x)
    x = self.conv_2(x) # 128
    x = self.norm_2(x)
    x = self.activation(x)
    x = self.conv_3(x) # 64
    x = self.norm_3(x)
    x = self.activation(x)

    return x # output Bx64x64x256

class Decoder(tf.keras.Model):
  def __init__(self):
    super(Decoder, self).__init__()
    # SpectralNormalization In decoder
    self.conv_1 = SpectralNormalization(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    self.conv_2 = SpectralNormalization(layers.Conv2D(128, (3, 3), strides=(1, 1), padding="same"))
    self.conv_3 = SpectralNormalization(layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same"))
    self.conv_4 = SpectralNormalization(layers.Conv2D(3, (3, 3), strides=(1, 1), padding="same", activation="sigmoid"))

    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.norm_3 = tfa.layers.InstanceNormalization()

    self.upsampling = layers.UpSampling2D(size=(2, 2))

    self.activation = layers.ReLU()
  
  def call(self, x):
    x = self.conv_1(x) # 256
    x = self.norm_1(x)
    x = self.activation(x)
    x = self.upsampling(x)
    x = self.conv_2(x) # 128
    x = self.norm_2(x)
    x = self.activation(x)
    x = self.upsampling(x)
    x = self.conv_3(x) # 64
    x = self.norm_3(x)
    x = self.activation(x)
    x = self.conv_4(x) # 3

    return x # output Bx256x256x3

class CreateMasks(tf.keras.Model):
  def __init__(self):
    super(CreateMasks, self).__init__()
    self.conv_m = layers.Conv2D(256, (3, 3), padding="same")
    self.conv_n = layers.Conv2D(256, (3, 3), padding="same")
  
  def call(self, inputs):
    target_distance_map = inputs[0]
    reference_distance = inputs[1]
    tensor_input = layers.Concatenate(axis=-1)([target_distance_map, reference_distance])

    m = self.conv_m(tensor_input)
    m = tf.keras.activations.sigmoid(m)

    n = self.conv_n(tensor_input)
    n = tf.keras.activations.sigmoid(n)

    return m, n

class LeftPart(tf.keras.Model):
  def __init__(self):
    super(LeftPart, self).__init__()
    kernels = 256 / 8
    self.conv = layers.Conv2D(kernels, (1, 1), padding="same")
  
  def call(self, inputs):
    target_distance_map = inputs[0]
    reference_distance_feat = inputs[1]
    reference_distance_x = self.conv(reference_distance_feat)
    target_distance_map_x = self.conv(target_distance_map)
    
    B, H, W, C = target_distance_map_x.shape


    reference_distance_x = layers.Reshape([H * W, C])(reference_distance_x)
    target_distance_map_x = layers.Reshape([H * W, C])(target_distance_map_x)
     
    M = tf.linalg.matmul(target_distance_map_x, reference_distance_x, transpose_b=True) #BxHWxHW
    # Multiplies each batch element separately

    return M

class RightPart(tf.keras.Model):
  def __init__(self):
    super(RightPart, self).__init__()
    kernels = 256 / 8
    self.conv = layers.Conv2D(kernels, (1, 1), padding="same")
  
  def call(self, inputs):
    m = inputs[0]
    reference_color = inputs[1]
    fm = reference_color * m # like attention

    x = self.conv(fm)
    
    B, H, W, C = x.shape

    x = layers.Reshape([H * W, C])(x) # BxHWxC
    x = tf.transpose(x, [0, 2, 1]) # BxCxHW

    return x, fm

class ColorTransformLayer(tf.keras.Model):
  def __init__(self):
    super(ColorTransformLayer, self).__init__()
    self.lp = LeftPart()
    self.rp = RightPart()
    self.get_masks = CreateMasks()
    self.conv = layers.Conv2D(256, (1, 1), padding="same")
  
  def call(self, inputs):
    target_distance_map = inputs[0]
    reference_distance_0 = inputs[1]
    reference_distance_1 = inputs[2]
    reference_color_0 = inputs[3]
    reference_color_1 = inputs[4]

    # target_distance_map, reference_distance_0, reference_distance_1, reference_color_0, reference_color_1
    B, H, W, _ = target_distance_map.shape

    M_0 = self.lp([target_distance_map, reference_distance_0]) #HWxHW
    M_1 = self.lp([target_distance_map, reference_distance_1]) #HWxHW

    matching_matrix = layers.Concatenate(axis=1)([M_0, M_1])
    matching_matrix = tf.keras.activations.softmax(matching_matrix) # HWKxHW

    small_m_0, n_0 = self.get_masks([target_distance_map, reference_distance_0])
    small_m_1, n_1 = self.get_masks([target_distance_map, reference_distance_1])

    c_0, fm_0 = self.rp([small_m_0, reference_color_0]) #BxCxHW
    c_1, fm_1 = self.rp([small_m_1, reference_color_1]) #BxCxHW

    reference_color_matrix = layers.Concatenate(axis=-1)([c_0, c_1])

    f_mat = tf.linalg.matmul(reference_color_matrix, matching_matrix) #BxCxHW
    _, C, _ = f_mat.shape

    f_mat = layers.Reshape([C, H, W])(f_mat)
    f_mat = tf.transpose(f_mat, [0, 2, 3, 1])

    f_mat = self.conv(f_mat) # BxHxWxC

    f_sim_left = (fm_1 * n_1) + ((n_1 - 1) * f_mat)
    f_sim_right = (fm_0 * n_0) + ((n_0 - 1) * f_mat)

    f_sim = (f_sim_left + f_sim_right) / 2
    # compute mean for each element in the batch

    return f_sim

class Embedder(tf.keras.Model):
  def __init__(self):
    super(Embedder, self).__init__()
    self.conv_1 = layers.Conv2D(64, (3, 3), strides=(1, 1), padding="same")
    self.conv_2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same")
    self.conv_3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same")
    self.conv_4 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")
    self.conv_5 = layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same")
  
  def call(self, inputs):
    reference_line_art = inputs[0]
    reference_color = inputs[1]
    # images not features
    x = layers.Concatenate(axis=-1)([reference_line_art, reference_color])

    x = self.conv_1(x) # 256
    x = self.conv_2(x) # 128
    x = self.conv_3(x) # 64
    x = self.conv_4(x) # 32
    x = self.conv_5(x) # 16

    x = layers.AveragePooling2D((16, 16))(x) # Bx1x1x512

    return x

class SEV(tf.keras.Model):
  def __init__(self):
    super(SEV, self).__init__()
    self.embedder = Embedder()
    self.dense_1 = layers.Dense(512)
    self.dense_2 = layers.Dense(512)
  
  def call(self, inputs):
    reference_line_art_0 = inputs[0]
    reference_color_0 = inputs[1] 
    reference_line_art_1 = inputs[2]
    reference_color_1 = inputs[3]

    latent_vector_0 = self.embedder([reference_line_art_0, reference_color_0])
    latent_vector_1 = self.embedder([reference_line_art_1, reference_color_1])

    x = (latent_vector_0 + latent_vector_1) / 2
    x = self.dense_1(x)
    x = self.dense_2(x)

    return x

class AdaInNormalization(tf.keras.layers.Layer):
  def __init__(self):
    super(AdaInNormalization, self).__init__()
    self.epsilon = 1e-5

  def call(self, x, style_vector):
    content_mean, content_variance = tf.nn.moments(x, [1, 2], keepdims=True) # Bx1x1xC
    content_sigma = tf.sqrt(tf.add(content_variance, self.epsilon))

    num_features = x.shape[-1]

    style_mean = style_vector[:, :, :, :num_features]
    style_sigma = style_vector[:, :, :, num_features:num_features*2]

    out = (x - content_mean) / content_sigma
    out = style_sigma * out + style_mean

    return out

class ResBlock(tf.keras.Model):
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv_1 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid')
    self.conv_2 = layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')
    self.conv_3 = layers.Conv2D(256, kernel_size=(1, 1), strides=(1, 1), padding='valid')

    self.AdaInLayer = AdaInNormalization()
  
  def call(self, inputs):
    x = inputs[0]
    style_vector = inputs[1]

    x_skip = x 

    x = self.conv_1(x)
    x = self.AdaInLayer(x, style_vector)
    x = layers.ReLU()(x)

    x = self.conv_2(x)
    x = self.AdaInLayer(x, style_vector)
    x = layers.ReLU()(x)

    x = self.conv_3(x)
    x = self.AdaInLayer(x, style_vector)

    x = layers.add([x, x_skip])
    x = layers.ReLU()(x)

    return x