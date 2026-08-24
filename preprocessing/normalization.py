"""
Physiological signal normalization.

This module implements signal normalization for ECG, trapezius EMG,
and GSR signals used in the multimodal physiological pain-recognition
framework.

Canonical normalization procedure
---------------------------------
For each physiological signal segment:

    1. Z-score standardization
       z = (x - mean) / standard_deviation

    2. Min-Max scaling
       x_norm = (z - z_min) / (z_max - z_min)

The resulting signal is scaled to the range [0, 1].

The functions in this module are dataset-independent and contain
no hard-coded input/output paths.

Author
------
Shreya
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.Series]


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_EPSILON = 1e-10

DEFAULT_SIGNAL_COLUMNS = (
    "ecg",
    "emg_trapezius",
    "gsr",
)


# ==========================================================
# INPUT CONVERSION
# ==========================================================

def _to_numpy(signal: ArrayLike) -> np.ndarray:
    """
    Convert a physiological signal to a one-dimensional NumPy array.

    Parameters
    ----------
    signal : array-like
        Input physiological signal.

    Returns
    -------
    numpy.ndarray
        One-dimensional floating-point array.

    Raises
    ------
    ValueError
        If the signal is empty or contains NaN/Inf values.
    """

    x = np.asarray(signal, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(
            "Physiological signal must be one-dimensional."
        )

    if x.size == 0:
        raise ValueError(
            "Physiological signal cannot be empty."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Physiological signal contains NaN or infinite values."
        )

    return x


# ==========================================================
# Z-SCORE STANDARDIZATION
# ==========================================================

def z_score(
    signal: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, float, float]:
    """
    Perform Z-score standardization.

    Parameters
    ----------
    signal : array-like
        Input physiological signal.
    epsilon : float, optional
        Threshold below which the signal is considered constant.

    Returns
    -------
    normalized : numpy.ndarray
        Z-score standardized signal.
    mean : float
        Mean of the original signal.
    std : float
        Standard deviation of the original signal.

    Notes
    -----
    Constant or near-constant signals are mapped to zero.
    """

    x = _to_numpy(signal)

    mean = float(np.mean(x))
    std = float(np.std(x))

    if std < epsilon:
        return np.zeros_like(x), mean, std

    normalized = (x - mean) / std

    return normalized, mean, std


# ==========================================================
# MIN-MAX NORMALIZATION
# ==========================================================

def minmax_scale(
    signal: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, float, float]:
    """
    Scale a signal to the range [0, 1].

    Parameters
    ----------
    signal : array-like
        Input physiological signal.
    epsilon : float, optional
        Threshold below which the signal range is considered constant.

    Returns
    -------
    normalized : numpy.ndarray
        Min-Max normalized signal.
    minimum : float
        Minimum value before scaling.
    maximum : float
        Maximum value before scaling.

    Notes
    -----
    Constant signals are mapped to zero.
    """

    x = _to_numpy(signal)

    minimum = float(np.min(x))
    maximum = float(np.max(x))

    signal_range = maximum - minimum

    if signal_range < epsilon:
        return np.zeros_like(x), minimum, maximum

    normalized = (
        (x - minimum) / signal_range
    )

    return normalized, minimum, maximum


# ==========================================================
# COMPLETE NORMALIZATION PIPELINE
# ==========================================================

def normalize_signal(
    signal: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Apply the complete normalization procedure.

    Processing
    ----------
    1. Z-score standardization.
    2. Min-Max scaling to [0, 1].

    Parameters
    ----------
    signal : array-like
        Input physiological signal.
    epsilon : float, optional
        Numerical threshold used for constant-signal detection.

    Returns
    -------
    normalized : numpy.ndarray
        Signal normalized to [0, 1].
    statistics : dict
        Normalization statistics containing original mean,
        original standard deviation, intermediate minimum,
        and intermediate maximum.
    """

    z_signal, mean, std = z_score(
        signal,
        epsilon=epsilon,
    )

    normalized, minimum, maximum = minmax_scale(
        z_signal,
        epsilon=epsilon,
    )

    statistics = {
        "mean": mean,
        "std": std,
        "z_min": minimum,
        "z_max": maximum,
    }

    return normalized, statistics


# ==========================================================
# QUALITY VALIDATION
# ==========================================================

def validate_normalized_signal(
    signal: ArrayLike,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, Union[bool, float]]:
    """
    Perform quality checks on a normalized physiological signal.

    Parameters
    ----------
    signal : array-like
        Normalized signal.
    epsilon : float, optional
        Numerical tolerance.

    Returns
    -------
    dict
        Signal quality statistics.
    """

    x = _to_numpy(signal)

    minimum = float(np.min(x))
    maximum = float(np.max(x))
    mean = float(np.mean(x))
    std = float(np.std(x))

    return {
        "has_nan": bool(np.isnan(x).any()),
        "has_infinite": bool(np.isinf(x).any()),
        "is_constant": bool(std < epsilon),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "std": std,
        "range_ok": bool(
            minimum >= -epsilon
            and maximum <= 1.0 + epsilon
        ),
    }


# ==========================================================
# AVAILABLE SIGNALS
# ==========================================================

def available_signals(
    df: pd.DataFrame,
    candidate_columns=DEFAULT_SIGNAL_COLUMNS,
    epsilon: float = DEFAULT_EPSILON,
) -> list[str]:
    """
    Identify usable physiological channels in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input physiological-signal DataFrame.
    candidate_columns : iterable of str, optional
        Candidate signal columns.
    epsilon : float, optional
        Minimum standard-deviation threshold.

    Returns
    -------
    list of str
        Names of available and non-constant signal channels.
    """

    signals = []

    for column in candidate_columns:

        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if values.size == 0:
            continue

        if not np.all(np.isfinite(values)):
            continue

        if np.std(values) < epsilon:
            continue

        signals.append(column)

    return signals


# ==========================================================
# DATAFRAME NORMALIZATION
# ==========================================================

def normalize_dataframe(
    df: pd.DataFrame,
    signal_columns=DEFAULT_SIGNAL_COLUMNS,
    copy: bool = True,
    epsilon: float = DEFAULT_EPSILON,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalize physiological channels in a DataFrame independently.

    Each available signal channel is normalized using:

        Z-score -> Min-Max scaling [0, 1]

    Parameters
    ----------
    df : pandas.DataFrame
        Input physiological signal segment.
    signal_columns : iterable of str, optional
        Signal columns to normalize.
    copy : bool, optional
        If True, operate on a copy of the input DataFrame.
    epsilon : float, optional
        Numerical threshold.

    Returns
    -------
    normalized_df : pandas.DataFrame
        DataFrame containing normalized physiological channels.
    statistics : dict
        Per-channel normalization statistics.

    Raises
    ------
    ValueError
        If a requested physiological channel contains NaN or Inf values.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    output = df.copy() if copy else df

    statistics = {}

    for column in signal_columns:

        if column not in output.columns:
            continue

        signal = pd.to_numeric(
            output[column],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        if signal.size == 0:
            continue

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                f"Channel '{column}' contains NaN or infinite values."
            )

        normalized, channel_stats = normalize_signal(
            signal,
            epsilon=epsilon,
        )

        output[column] = normalized

        quality = validate_normalized_signal(
            normalized,
            epsilon=epsilon,
        )

        statistics[column] = {
            **channel_stats,
            **quality,
        }

    return output, statistics
