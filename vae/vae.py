from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model
from keras.losses import binary_crossentropy, mse
from tensorflow.keras import backend as K


def get_models(inputs_shape, latent_dim, filters):
    #region Encoder

    #region 256x256 -> 64x64
    i       = Input(shape=inputs_shape, name="encoder_input")
    cx      = Conv2D(filters=filters,
                     kernel_size=9,
                     padding="same",
                     activation="relu",
                     use_bias=False)(i)
    cx      = MaxPooling2D(pool_size=(4, 4), padding="same")(cx)
    cx      = BatchNormalization()(cx)
    #endregion

    #region 64x64 -> 32x32
    cx      = Conv2D(filters=filters,
                     kernel_size=5,
                     padding="same",
                     activation="relu",
                     use_bias=False)(cx)
    cx      = MaxPooling2D(pool_size=(2, 2), padding="same")(cx)
    cx      = BatchNormalization()(cx)
    #endregion

    #region 32x32 -> 16x16
    cx      = Conv2D(filters=filters,
                     kernel_size=5,
                     padding="same",
                     activation="relu",
                     use_bias=False)(cx)
    cx      = MaxPooling2D(pool_size=(2, 2), padding="same")(cx)
    cx      = BatchNormalization()(cx)
    #endregion

    x       = Flatten()(cx)
    x       = Dense(256, activation="relu")(x)
    x       = BatchNormalization()(x)
    mu      = Dense(latent_dim, name="latent_mu")(x)
    sigma   = Dense(latent_dim, name="latent_sigma")(x)

    conv_shape = K.int_shape(cx)

    #region Reparameterisation
    def sample_z(args):
        mu, sigma = args
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        eps = K.random_normal(shape=(batch, dim))
        return mu + K.exp(sigma / 2) * eps
    #endregion

    z       = Lambda(sample_z, output_shape=(latent_dim, ), name="z")([mu, sigma])
    encoder = Model(i, [mu, sigma, z], name="encoder")

    #endregion

    #region Decoder
    d_i     = Input(shape=(latent_dim, ), name="decoder_output")
    x       = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3], activation="relu")(d_i)
    x       = BatchNormalization()(x)
    x       = Reshape((conv_shape[1] * conv_shape[2] * conv_shape[3]))(x)

    #region 16x16 -> 32x32
    cx      = Conv2DTranspose(filters=filters,
                              kernel_size=9,
                              strides=2,
                              padding="same",
                              activation="relu",
                              use_bias=False)(x)
    cx      = BatchNormalization()(cx)
    #endregion

    #region 32x32 -> 64x64
    cx      = Conv2DTranspose(filters=filters,
                              kernel_size=5,
                              strides=2,
                              padding="same",
                              activation="relu",
                              use_bias=False)(cx)
    cx      = BatchNormalization()(cx)
    #endregion

    #region 64x64 -> 256x256
    o       = Conv2DTranspose(filters=inputs_shape[2],
                              kernel_size=3,
                              strides=2,
                              padding="same",
                              activation="sigmoid",
                              use_bias=False)(cx)
    #endregion

    decoder = Model(d_i, o, name="decoder")

    #endregion

    vae_outputs = decoder(encoder(i)[2])
    vae         = Model(i, vae_outputs, name="vae")
    return encoder, decoder, vae


class VariationalAutoencoder(keras.Model):

    def __init__(self, filters, inputs_shape=(256, 256, 1)):
        super(VariationalAutoencoder, self).__init__()
        self.inputs_shape = inputs_shape
        self.filters = filters
        self.enc, self.dec, self.vae = get_models(self.inputs_shape, self.filters)

    def compile(self, optimizer, loss=None):
        super(VariationalAutoencoder, self).compile()
        self.loss_fn = loss
        self.optimizer = optimizer

    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            encoded = self.enc(x)
            decoded = self.dec(encoded[2])
            loss    = self.loss_fn(x, x)
        grads = tape.gradient(loss, self.encoder.trainable_weights)
        self.optimizer.apply_gradients()

