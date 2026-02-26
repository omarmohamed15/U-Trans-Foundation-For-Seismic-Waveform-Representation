# utrans/encoder.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout, Flatten, Reshape,
    BatchNormalization, Activation, Add, concatenate
)
import numpy as np


# =============================================================================
# Encoder Components (CCT + UNET) Used in U-Trans
# =============================================================================
#
# This module contains the core building blocks used inside U-Trans:
#
# 1) A Compact Convolutional Transformer (CCT) block used as an attention-based
#    bottleneck ("create_cct_model1").
#
# 2) A 1D UNET backbone ("UNET") that uses the CCT bottleneck:
#       drop5 = create_cct_model1(conv5)
#
# Notes:
# - These functions/classes should match the exact implementation used during
#   training to ensure pretrained weights load correctly.
# - This file only defines the architecture blocks. Weight loading and feature
#   extraction utilities are implemented in `utrans/foundation.py`.
# =============================================================================


# -----------------------------------------------------------------------------
# CCT (Transformer) Hyperparameters
# -----------------------------------------------------------------------------
stochastic_depth_rate = 0.1
positional_emb = False
conv_layers = 1

# The following are defined in your original code. Some (like input_shape) are
# not directly used in create_cct_model1(), but kept for completeness.
num_classes = 1
input_shape = (375, 256)

projection_dim = 80
num_heads = 4
transformer_units = [projection_dim, projection_dim]
transformer_layers = 4


# -----------------------------------------------------------------------------
# Helper: MLP block used inside the Transformer
# -----------------------------------------------------------------------------
def mlp(x, hidden_units, dropout_rate):
    """
    Feed-forward network used inside each Transformer block.

    Parameters
    ----------
    x : Tensor
        Input token tensor.
    hidden_units : list[int]
        Dense layer widths.
    dropout_rate : float
        Dropout rate.

    Returns
    -------
    Tensor
        Transformed tensor (same sequence length, updated channel dim).
    """
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# -----------------------------------------------------------------------------
# Stochastic Depth (DropPath) layer
# -----------------------------------------------------------------------------
class StochasticDepth(layers.Layer):
    """
    Implements Stochastic Depth (a.k.a. DropPath) for residual branches.

    During training, the residual branch is randomly dropped with probability
    `drop_prob`. During inference, the input is returned unchanged.
    """
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = float(drop_prop)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_prop": self.drop_prob})
        return cfg

    def call(self, x, training=None):
        if training:
            keep_prob = 1.0 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


# -----------------------------------------------------------------------------
# CCT Tokenizer (Conv1D projection to token sequence)
# -----------------------------------------------------------------------------
class CCTTokenizer1(layers.Layer):
    """
    Tokenizer used in create_cct_model1().

    It applies a small Conv1D stack to convert UNET bottleneck features into a
    token sequence used by the Transformer block.
    """
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=(1, 1, 1, 1),
        num_conv_layers=conv_layers,
        num_output_channels=None,
        positional_emb=positional_emb,
        **kwargs,
    ):
        super(CCTTokenizer1, self).__init__(**kwargs)

        if num_output_channels is None:
            num_output_channels = [int(projection_dim)] * 8

        self.conv_model = tf.keras.Sequential(name="cct_tokenizer_conv")
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv1D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="same",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )

        self.positional_emb = positional_emb

    def call(self, images):
        """
        Parameters
        ----------
        images : Tensor
            Shape typically (B, T, C)

        Returns
        -------
        Tensor
            Token features (B, T, projection_dim)
        """
        outputs = self.conv_model(images)
        # `reshaped` is unused in your original code; kept for clarity.
        _reshaped = tf.reshape(outputs, (-1, tf.shape(outputs)[1], tf.shape(outputs)[2]))
        return outputs

    def positional_embedding(self, image_size):
        """
        Optional positional embedding helper (disabled by default).
        """
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, 1))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = int(dummy_outputs.shape[1])
            proj_dim = int(dummy_outputs.shape[-1])
            embed_layer = layers.Embedding(input_dim=sequence_length, output_dim=proj_dim)
            return embed_layer, sequence_length
        return None


# -----------------------------------------------------------------------------
# CCT Model Block (Transformer encoder)
# -----------------------------------------------------------------------------
def create_cct_model1(inputs):
    """
    Compact Convolutional Transformer (CCT) block.

    This function:
    1) Tokenizes the input with Conv1D (CCTTokenizer1)
    2) Applies `transformer_layers` Transformer blocks
    3) Returns the final token representation (LayerNorm applied)

    Parameters
    ----------
    inputs : Tensor
        UNET bottleneck tensor, typically shape (B, T', C')

    Returns
    -------
    Tensor
        Transformer token representation with same sequence length:
        shape ~ (B, T', projection_dim)
    """
    cct_tokenizer = CCTTokenizer1()
    encoded_patches = cct_tokenizer(inputs)

    # Positional embedding is optional (disabled by default)
    if positional_emb:
        # NOTE: `image_size` was referenced in the original code but not defined
        # in this file. If you enable positional_emb, define image_size properly.
        raise ValueError("positional_emb=True is not supported unless image_size is defined.")

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    return representation


# -----------------------------------------------------------------------------
# convF1 Residual Conv Block
# -----------------------------------------------------------------------------
def convF1(inpt, D1, fil_ord, Dr):
    """
    Residual Conv1D block used in U-Trans (convF1).

    Parameters
    ----------
    inpt : Tensor
        Input tensor.
    D1 : int
        Output channels of the final Conv1D in the block.
    fil_ord : int
        Kernel size.
    Dr : float
        Dropout rate.

    Returns
    -------
    Tensor
        Output tensor after residual Conv1D operations.
    """
    filters = int(inpt.shape[-1])

    pre = Conv1D(filters, fil_ord, padding="same", kernel_initializer="he_normal")(inpt)
    pre = BatchNormalization()(pre)
    pre = Activation(tf.nn.gelu)(pre)

    inf = Conv1D(filters, fil_ord, padding="same", kernel_initializer="he_normal")(pre)
    inf = BatchNormalization()(inf)
    inf = Activation(tf.nn.gelu)(inf)
    inf = Add()([inf, inpt])

    inf1 = Conv1D(D1, fil_ord, padding="same", kernel_initializer="he_normal")(inf)
    inf1 = BatchNormalization()(inf1)
    inf1 = Activation(tf.nn.gelu)(inf1)

    return Dropout(Dr)(inf1)


# -----------------------------------------------------------------------------
# UNET Backbone with CCT Bottleneck
# -----------------------------------------------------------------------------
def UNET(inputs, D1):
    """
    1D UNET backbone used in U-Trans.

    The bottleneck uses the CCT block:
        drop5 = create_cct_model1(conv5)

    Parameters
    ----------
    inputs : Tensor
        Input waveform tensor, shape (B, 6000, 3)
    D1 : int
        Base channel width.

    Returns
    -------
    Tensor
        Final UNET feature map (B, 6000, D1)
    """
    D2 = int(D1 * 2)
    D3 = int(D2 * 2)
    D4 = int(D3 * 2)
    D5 = int(D4 * 2)

    # Encoder
    conv1 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(inputs)
    conv1 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool1)
    conv2 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool2)
    conv3 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool3)
    conv4 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    conv44 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool4)
    conv44 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv44)
    pool44 = MaxPooling1D(pool_size=5)(conv44)

    # Bottleneck
    conv5 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(pool44)
    conv5 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv5)

    # CCT bottleneck
    drop5 = create_cct_model1(conv5)

    # Decoder
    up66 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=5)(drop5))
    merge66 = concatenate([pool4, up66], axis=-1)
    conv66 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge66)
    conv66 = Conv1D(D5, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv66)

    up6 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv66))
    merge6 = concatenate([conv4, up6], axis=-1)
    conv6 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge6)
    conv6 = Conv1D(D4, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv6)

    up7 = Conv1D(D3, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv6))
    merge7 = concatenate([conv3, up7], axis=-1)
    conv7 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge7)
    conv7 = Conv1D(D3, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv7)

    up8 = Conv1D(D2, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv7))
    merge8 = concatenate([conv2, up8], axis=-1)
    conv8 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge8)
    conv8 = Conv1D(D2, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv8)

    up9 = Conv1D(D1, 2, activation="relu", padding="same", kernel_initializer="he_normal")(UpSampling1D(size=2)(conv8))
    merge9 = concatenate([conv1, up9], axis=-1)
    conv9 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(merge9)
    conv9 = Conv1D(D1, 3, activation="relu", padding="same", kernel_initializer="he_normal")(conv9)

    return conv9