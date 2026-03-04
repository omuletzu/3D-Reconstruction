import tensorflow as tf
from tensorflow.keras import layers, models, initializers, metrics

class AddCoords(layers.Layer):
  def call(self, input_tensor):
    batch_shape = tf.shape(input_tensor)

    batch_size = batch_shape[0]
    x_dim = batch_shape[1]
    y_dim = batch_shape[2]

    x_range = tf.range(x_dim, dtype='float32')
    y_range = tf.range(y_dim, dtype='float32')

    x_channel = (x_range / tf.cast(x_dim - 1, 'float32')) * 2 - 1
    y_channel = (y_range / tf.cast(y_dim - 1, 'float32')) * 2 - 1

    x_grid, y_grid = tf.meshgrid(x_channel, y_channel, indexing='ij')

    x_grid = tf.expand_dims(x_grid, axis=-1)
    y_grid = tf.expand_dims(y_grid, axis=-1)

    x_grid = tf.expand_dims(x_grid, axis=0)
    y_grid = tf.expand_dims(y_grid, axis=0)

    x_channel = tf.tile(x_grid, [batch_size, 1, 1, 1])
    y_channel = tf.tile(y_grid, [batch_size, 1, 1, 1])

    return tf.concat([input_tensor, x_channel, y_channel], axis=-1)

class GeMPool(layers.Layer):
  def __init__(self, p=3.0, eps=1e-6, **kwargs):
    super(GeMPool, self).__init__(**kwargs)
    self.init_p = p
    self.eps = eps

  def build(self, input_shape):
    self.p = self.add_weight(
        name='p',
        shape=(1,),
        initializer=initializers.Constant(self.init_p),
        trainable=True
    )

    super(GeMPool, self).build(input_shape)

  def call(self, x):
    x = tf.maximum(x, self.eps)
    x = tf.pow(x, self.p)
    x = tf.reduce_mean(x, axis=[1, 2])
    return tf.pow(x, 1.0 / self.p)

class PositionalEmbedding(layers.Layer):
  def build(self, input_shape):
    self.emb_pos = self.add_weight(
        name='emb_pos',
        shape=(1, input_shape[1], input_shape[2]),
        trainable=True
    )

  def call(self, x):
    return x + self.emb_pos

def build_model(input_shape=(32, 32, 1)):
  input_layer = layers.Input(shape=input_shape, name='input_layer')

  x = AddCoords(name='coord_conv')(input_layer)

  x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', use_bias=False)(x)
  x = layers.BatchNormalization(scale=False)(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_uniform', use_bias=False)(x)
  x = layers.BatchNormalization(scale=False)(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(64, 3, padding='same', strides=2, kernel_initializer='he_uniform', use_bias=False)(x)
  x = layers.BatchNormalization(scale=False)(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(64, 3, padding='same', kernel_initializer='he_uniform', use_bias=False)(x)
  x = layers.BatchNormalization(scale=False)(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(128, 3, padding='same', strides=2, kernel_initializer='he_uniform', use_bias=False)(x)
  x = layers.BatchNormalization(scale=False)(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(128, 3, padding='same', kernel_initializer='he_uniform', use_bias=False)(x)
  x = layers.BatchNormalization(scale=False)(x)
  x = layers.Activation('relu')(x)

  x = layers.Dropout(0.3)(x)

  x_flat = layers.Reshape((64, 128), name="flatten_sequence")(x)

  x_pos = PositionalEmbedding(name='pos_embedding')(x_flat)

  attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x_pos, x_pos)

  x_trans = layers.Add()([x_pos, attention_output])
  x_trans = layers.LayerNormalization(epsilon=1e-6)(x_trans)

  dense_output = layers.Dense(256, activation='relu')(x_trans)
  dense_output = layers.Dense(128, activation='relu')(dense_output)

  x_trans = layers.Add()([x_trans, dense_output])
  x_trans = layers.LayerNormalization(epsilon=1e-6)(x_trans)

  x_spatial = layers.Reshape((8, 8, 128), name='restore_layer')(x_trans)

  x_gem = GeMPool(name='gem_pooling_layer')(x_spatial)

  output = layers.Dense(128, name='final_dense_layer')(x_gem)
  output = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), output_shape=(128,), name='descriptor')(output)

  model = models.Model(inputs=input_layer, outputs=output, name='feature_model')

  return model

class SiameseModel(models.Model):
  def __init__(self, base_network, margin=1.0, **kwargs):
    super(SiameseModel, self).__init__(**kwargs)
    self.base_network = base_network
    self.margin = margin
    self.loss_tracker = metrics.Mean(name='Loss')
    self.accuracy_tracker = metrics.Mean(name='accuracy')
    self.pos_dist_tracker = metrics.Mean(name='dist_pos')
    self.neg_dist_tracker = metrics.Mean(name='dist_neg')

  def call(self, inputs):
    return self.base_network(inputs)

  def compute_loss_and_metrics(self, data):
    (anchor, pos, neg), _ = data

    anchor_output = self.base_network(anchor, training=True)
    pos_output = self.base_network(pos, training=True)
    neg_output = self.base_network(neg, training=True)

    pos_dist = tf.reduce_sum(tf.square(anchor_output - pos_output), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor_output - neg_output), axis=1)

    loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
    loss = tf.reduce_mean(loss)

    is_correct = pos_dist < neg_dist
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return loss, accuracy, tf.reduce_mean(pos_dist), tf.reduce_mean(neg_dist)

  def train_step(self, data):
    with tf.GradientTape() as tape:
      loss, acc, pos_dist_mean, neg_dist_mean = self.compute_loss_and_metrics(data)

    grads = tape.gradient(loss, self.base_network.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.base_network.trainable_weights))

    self.loss_tracker.update_state(loss)
    self.accuracy_tracker.update_state(acc)
    self.pos_dist_tracker.update_state(pos_dist_mean)
    self.neg_dist_tracker.update_state(neg_dist_mean)

    return {m.name: m.result() for m in self.metrics}

  def test_step(self, data):
    loss, acc, pos_dist_mean, neg_dist_mean = self.compute_loss_and_metrics(data)

    self.loss_tracker.update_state(loss)
    self.accuracy_tracker.update_state(acc)
    self.pos_dist_tracker.update_state(pos_dist_mean)
    self.neg_dist_tracker.update_state(neg_dist_mean)

    return {m.name: m.result() for m in self.metrics}

  @property
  def metrics(self):
      return [self.loss_tracker, self.accuracy_tracker, self.pos_dist_tracker, self.neg_dist_tracker]