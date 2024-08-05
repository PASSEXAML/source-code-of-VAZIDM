from keras.engine.topology import Layer
from keras.layers import Lambda, Dense
from keras.engine.base_layer import InputSpec
from keras import backend as K
import tensorflow as tf


import tensorflow as tf

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, input_size, num_heads, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_heads = num_heads
        self.depth = input_size // num_heads
        self.query_dense = tf.keras.layers.Dense(units=input_size)
        self.key_dense = tf.keras.layers.Dense(units=input_size)
        self.value_dense = tf.keras.layers.Dense(units=input_size)

    def split_heads(self, x):
        batch_size = tf.shape(x)[0]  # 使用 tf.shape 以适应动态批量大小
        x = tf.reshape(x, (batch_size, self.num_heads, self.depth))
        return x

    def call(self, inputs):
        query = self.split_heads(self.query_dense(inputs))
        key = self.split_heads(self.key_dense(inputs))
        value = self.split_heads(self.value_dense(inputs))

        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        attention_output = tf.matmul(attention_scores, value)

        attention_output = tf.reshape(attention_output, [tf.shape(inputs)[0], self.input_size])
        return attention_output


# class SelfAttention(tf.keras.layers.Layer):
#     def __init__(self, input_size, num_heads, **kwargs):
#         super(SelfAttention, self).__init__(**kwargs)
#         self.input_size = input_size
#         self.num_heads = num_heads
#         self.depth = input_size // num_heads
#
#         # 为Query, Key, Value创建Dense层
#         self.query_dense = tf.keras.layers.Dense(units=input_size)
#         self.key_dense = tf.keras.layers.Dense(units=input_size)
#         self.value_dense = tf.keras.layers.Dense(units=input_size)
#
#     def split_heads(self, x):
#         # print("x", x)
#         # 重排x到 (batch_size, num_heads, seq_len, depth)
#         x_shape = tf.shape(x)
#         x = tf.reshape(x, (x_shape[0], 1, self.num_heads, self.depth))
#         print("x_shape", x.shape)
#         return tf.transpose(x, perm=[0, 2, 1, 3])
#
#     def call(self, inputs):
#         query = self.split_heads(self.query_dense(inputs))
#         key = self.split_heads(self.key_dense(inputs))
#         value = self.split_heads(self.value_dense(inputs))
#
#         # 计算attention
#         attention_scores = tf.matmul(query, key, transpose_b=True)
#         attention_scores = tf.nn.softmax(attention_scores, axis=-1)
#         attention_output = tf.matmul(attention_scores, value)
#         print("a_shape", attention_output.shape)
#         # 重排回原始形状
#         attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
#         print("a1_shape", attention_output.shape)
#         output_shape = tf.shape(attention_output)
#         print("o_shape", output_shape.shape)
#         attention_output = tf.reshape(attention_output, (output_shape[0], attention_output.shape[-1]))
#         return attention_output

class ConstantDispersionLayer(Layer):
    '''
        An identity layer which allows us to inject extra parameters
        such as dispersion to Keras models
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(shape=(1, input_shape[1]),
                                     initializer='zeros',
                                     trainable=True,
                                     name='theta')
        self.theta_exp = tf.clip_by_value(K.exp(self.theta), 1e-3, 1e4)
        super().build(input_shape)

    def call(self, x):
        return tf.identity(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class SliceLayer(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]

class ElementwiseDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert (input_dim == self.units) or (self.units == 1), \
               "Input and output dims are not compatible"

        # shape=(input_units, ) makes this elementwise bcs of broadcasting
        self.kernel = self.add_weight(shape=(self.units,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        # use * instead of tf.matmul, we need broadcasting here
        output = inputs * self.kernel
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

nan2zeroLayer = Lambda(lambda x: tf.where(tf.is_nan(x), tf.zeros_like(x), x))
ColwiseMultLayer = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)))