import os
import pickle
import numpy as np
import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda, LeakyReLU
from keras.models import Model
from keras.regularizers import l1_l2
from keras.objectives import mean_squared_error
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
from .datahandle import write_text_matrix
from .layers import ConstantDispersionLayer, SliceLayer, ColwiseMultLayer, ElementwiseDense, SelfAttention
from .loss import NB, ZINB

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
advanced_activations = ('PReLU', 'LeakyReLU')



class VariationalAutoencoder():
    def __init__(self,
                input_size,
                hidden_size=(64, 32, 64),
                output_size=None,
                l1_coef=0.001,
                # l1_coef=0.,
                l2_coef=0.,
                l2_enc_coef=0.,
                l1_enc_coef=0.001,
                ridge=0.,
                hidden_dropout=0.01,
                input_dropout=0.,
                batchnorm=True,
                init='glorot_uniform',
                file_path=None,
                activation='relu',
                debug=True,
                z_mean=None,
                z_log_var=None,
                ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.l2_enc_coef = l2_enc_coef
        self.l1_enc_coef = l1_enc_coef
        self.ridge = ridge
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.init = init
        self.loss = {}
        self.file_path = file_path
        self.extra_models = {}
        self.model = None
        self.encoder = None
        self.decoder = None
        self.activation = activation
        self.input_layer = None
        self.debug = debug
        self.z_mean = z_mean
        self.z_log_var = z_log_var

        if self.output_size is None:
            self.output_size = self.input_size

        if isinstance(self.hidden_size, list):
            assert len(self.hidden_size) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout] * len(self.hidden_size)

    def save(self):
        if self.file_path:
            os.makedirs(self.file_path, exist_ok=True)
            with open(os.path.join(self.file_path, 'model.pickle'), 'wb') as f:
                pickle.dump(self, f)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def build(self):

        self.input_layer = Input(shape=(self.input_size, ))
        last_hidden = self.input_layer
        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            last_hidden = Dense(hid_size, activation=self.activation)(last_hidden)
            if hid_drop > 0.0:
                last_hidden = Dropout(hid_drop)(last_hidden)

        self.z_mean = Dense(self.hidden_size[-1], name='z_mean')(last_hidden)
        self.z_log_var = Dense(self.hidden_size[-1], name='z_log_var')(last_hidden)
        z = Lambda(self.sampling, output_shape=(self.hidden_size[-1],), name='z')([self.z_mean, self.z_log_var])

        for hid_size in reversed(self.hidden_size[:-1]):
            z = Dense(hid_size, activation=self.activation)(z)

        self.decoder_output = Dense(self.output_size, activation='sigmoid')(z)

        self.build_output()

    def build_output(self):

        pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(self.decoder_output)

        disp = Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                           name='dispersion')(self.decoder_output)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.decoder_output)

        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)

        self.loss = zinb.loss
        output = mean
        output = SliceLayer(0, name='slice')([output, disp, pi])
        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=self.input_layer, outputs=output)


    def predict(self, adata, mode='denoise', return_info=True, copy=False):
        assert mode in ('denoise', 'latent', 'full'), 'Unknown mode'
        adata = adata.copy() if copy else adata

        colnames = adata.columns
        rownames = adata.index

        if return_info:
            os.makedirs(self.file_path, exist_ok=True)
            file_path = self.file_path
            output_values = self.model.predict(adata.values)
            write_text_matrix(output_values,
                              os.path.join(file_path, 'output_values.csv'),
                              rownames=rownames, colnames=colnames, transpose=True)

class GAN():
    def __init__(self, latent_dim=100, input_size=None):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

    def build_generator(self):
        noise = Input(shape=(self.latent_dim,))

        x = Dense(256)(noise)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)

        x = Dense(1024)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        return Model(noise, x)

    def build_discriminator(self):
        data = Input(shape=(self.input_size,))

        x = Dense(512)(data)
        x = LeakyReLU(0.2)(x)

        x = Dense(256)(x)
        x = LeakyReLU(0.2)(x)

        x = Dense(self.input_size, activation='sigmoid')(x)
        return Model(data, x)

class GAN_VAE(VariationalAutoencoder):
    def __init__(self, input_size, hidden_size=(64, 32, 64), output_size=None, latent_dim=100, **kwargs):
        super(GAN_VAE, self).__init__(input_size, hidden_size, output_size, **kwargs)
        self.gan = GAN(latent_dim, input_size=self.input_size)
        self.decoder = self.gan.generator

    def build(self):
        super(GAN_VAE, self).build()
        self.model = Model(inputs=self.input_layer, outputs=self.gan.discriminator(self.decoder_output))


VAE_types = {'normal': VariationalAutoencoder, 'GAN_VAE': GAN_VAE}