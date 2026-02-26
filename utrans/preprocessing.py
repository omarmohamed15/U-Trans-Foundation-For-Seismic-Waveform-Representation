# utrans/preprocessing.py

import numpy as np

# =============================================================================
# Preprocessing Utilities for U-Trans
# =============================================================================
#
# This module provides lightweight NumPy-based preprocessing helpers used by
# U-Trans example notebooks and downstream task pipelines.
#
# The U-Trans foundation model expects input waveforms with shape:
#     (B, 6000, 3)
#
# Where:
#   - B     : batch size (number of waveform windows)
#   - 6000  : samples per window (fixed-length input)
#   - 3     : three-component waveform (e.g., E/N/Z or N/E/Z)
#
# These functions are intentionally minimal and framework-agnostic:
# - ensure_shape(...) validates waveform tensor dimensions before inference
# - normalize_std(...) performs per-trace standardization (channel-wise)
# =============================================================================


def ensure_shape(x, length=6000, channels=3):
    """
    Validate input waveform shape.

    Parameters
    ----------
    x : array-like
        Input waveforms as a NumPy array (or convertible to one).
        Expected shape: (B, L, C)
    length : int, default=6000
        Expected number of time samples per waveform window.
    channels : int, default=3
        Expected number of waveform channels/components.

    Returns
    -------
    x : np.ndarray
        Input converted to NumPy array (float or original dtype preserved),
        guaranteed to have shape (B, length, channels).

    Raises
    ------
    ValueError
        If input does not have 3 dimensions or the expected (L, C).
    """
    x = np.asarray(x)

    if x.ndim != 3:
        raise ValueError(f"Expected (B, L, C), got shape {x.shape}")

    if x.shape[1] != length or x.shape[2] != channels:
        raise ValueError(f"Expected (B, {length}, {channels}), got {x.shape}")

    return x


def normalize_std(x, eps=1e-8):
    """
    Per-trace standardization over time for each channel.

    Standardizes each waveform independently:
        x_norm = (x - mean_t) / (std_t + eps)

    Parameters
    ----------
    x : np.ndarray
        Waveform batch with shape (B, L, C).
    eps : float, default=1e-8
        Small constant for numerical stability.

    Returns
    -------
    x_norm : np.ndarray
        Standardized waveforms with the same shape as x: (B, L, C)

    Notes
    -----
    - Mean and standard deviation are computed across the time axis (axis=1),
      independently for each trace and channel.
    - This is typically used before passing waveforms to the U-Trans encoder.
    """
    x = np.asarray(x)

    mu = x.mean(axis=1, keepdims=True)   # (B, 1, C)
    sd = x.std(axis=1, keepdims=True)    # (B, 1, C)

    return (x - mu) / (sd + eps)