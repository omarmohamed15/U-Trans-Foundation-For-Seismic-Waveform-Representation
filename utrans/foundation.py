# utrans/foundation.py

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Flatten, Reshape
)

from .layers import *  # expects: UNET(...), convF1(...), and any required custom layers


# =============================================================================
# U-Trans Foundation Model Utilities
# =============================================================================
#
# This module provides a lightweight interface to:
#
# 1) Rebuild the original U-Trans UNET-based foundation model architecture
# 2) Load pretrained weights into the rebuilt model
# 3) Extract:
#    - Latent transformer representation (from an internal layer)
#    - Time-aligned decoder features (6000,1) ready for concatenation
#
# Notes:
# - The latent representation is currently extracted from `modeloriginal.layers[63]`.
#   This index corresponds to a transformer layer output with shape (B, 75, 80)
#   in your trained checkpoint, where 75*80 = 6000.
# - If your architecture changes, the layer index may shift. In that case, either:
#   (a) update the index, or
#   (b) switch to name-based / shape-based layer selection for robustness.
# =============================================================================


# -----------------------------------------------------------------------------
# Build the full UNET-based foundation model (same structure used in training)
# -----------------------------------------------------------------------------
def build_modeloriginal(input_shape=(6000, 3), D1=5):
    """
    Builds the original UNET-based model used in U-Trans training.

    Parameters
    ----------
    input_shape : tuple
        Input waveform shape (L, C). Default: (6000, 3)
    D1 : int
        Base UNET filter multiplier. Default: 5

    Returns
    -------
    modeloriginal : tf.keras.Model
        Full model with output head "picker_PP" (3 channels).
    """
    inp = Input(shape=input_shape, name="input")

    # UNET backbone (must be provided in .layers import)
    conv1 = UNET(inp, D1)

    # Final convs used in the original model definition
    out = Conv1D(D1, 3, padding="same", kernel_initializer="he_normal")(conv1)
    out = Conv1D(3, 3, padding="same", kernel_initializer="he_normal", name="picker_PP")(out)

    modeloriginal = Model(inp, out, name="UTrans_UNET_full")
    return modeloriginal


def load_modeloriginal(unet_weights_path, input_shape=(6000, 3), D1=5):
    """
    Builds + loads pretrained weights into the original U-Trans model.

    Parameters
    ----------
    unet_weights_path : str
        Path to pretrained weights file (.h5)
    input_shape : tuple
        Input waveform shape (L, C). Default: (6000, 3)
    D1 : int
        Base UNET filter multiplier. Default: 5

    Returns
    -------
    modeloriginal : tf.keras.Model
        Model with loaded weights, ready for feature extraction.
    """
    modeloriginal = build_modeloriginal(input_shape=input_shape, D1=D1)

    # Compile is not strictly required for inference, but kept for consistency.
    modeloriginal.compile(optimizer="adam", loss="binary_crossentropy")

    modeloriginal.load_weights(unet_weights_path)
    return modeloriginal


# -----------------------------------------------------------------------------
# Feature extraction: latent representation + decoder features
# -----------------------------------------------------------------------------
def get_latent_model(unet_weights_path, input_shape=(6000, 3), D1=5):
    """
    Returns a model that outputs the latent transformer representation.

    Output
    ------
    latent : Tensor
        Shape: (B, 75, 80) in the provided checkpoint.
        This is extracted from an internal layer of the pretrained model.

    Notes
    -----
    The layer index (63) was identified from your checkpoint inspection:
        modeloriginal.layers[63].output_shape == (None, 75, 80)
    """
    modeloriginal = load_modeloriginal(unet_weights_path, input_shape=input_shape, D1=D1)

    # Internal latent representation (transformer tokens)
    latent = modeloriginal.layers[63].output

    return Model(modeloriginal.input, latent, name="UTrans_latent")


def get_decoder_model(unet_weights_path, input_shape=(6000, 3), D1=5):
    """
    Returns a model that outputs decoder features aligned with the input time axis.

    Pipeline
    --------
    waveform -> latent (B, 75, 80) -> convF1 x3 -> Flatten -> Reshape((6000,1))

    Output
    ------
    decoder_features : Tensor
        Shape: (B, 6000, 1)
        Ready to concatenate with downstream task inputs (e.g., picking/detection).

    Returns
    -------
    decoder_model : tf.keras.Model
        waveform -> decoder_features
    decoder_features : Tensor
        The output tensor (useful for debugging and shape inspection)
    """
    modeloriginal = load_modeloriginal(unet_weights_path, input_shape=input_shape, D1=D1)

    latent = modeloriginal.layers[63].output  # expected shape: (B, 75, 80)

    # Decoder head: converts token representation into time-aligned feature stream
    x = convF1(latent, 80, 13, 0.1)
    x = convF1(x, 80, 13, 0.1)
    x = convF1(x, 80, 13, 0.1)

    # Flatten (75*80 = 6000) then reshape to (6000,1)
    x = Flatten()(x)
    x = Reshape((6000, 1), name="decoder_features")(x)

    decoder_model = Model(modeloriginal.input, x, name="UTrans_decoder_features")
    return decoder_model, x