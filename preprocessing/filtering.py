"""
Physiological signal filtering.

This module implements the preprocessing filters used for ECG,
trapezius EMG, and GSR signals in the multimodal physiological
pain-recognition framework.

Canonical preprocessing settings
--------------------------------
Sampling frequency : 512 Hz

ECG:
    Band-pass : 0.5–45 Hz
    Notch     : 50 Hz

Trapezius EMG:
    Band-pass : 20–250 Hz
    Notch     : 50 Hz

GSR:
    Low-pass  : 0.5 Hz

The functions in this module are dataset-independent and contain
no hard-coded input/output paths.

Author
------
Shreya
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch


ArrayLike = Union[np.ndarray, pd.Series]


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_FS = 512

DEFAULT_NOTCH_FREQ = 50.0
DEFAULT_NOTCH_Q = 30.0

ECG_LOWCUT = 0.5
ECG_HIGHCUT = 45.0

EMG_LOWCUT = 20.0
EMG_HIGHCUT = 250.0

GSR_CUTOFF = 0.5

DEFAULT_FILTER_ORDER = 4


# ==========================================================
# VALIDATION
# ==========================================================

def validate_sampling_frequency(fs: float) -> None:
    """
    Validate sampling frequency.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.

    Raises
    ------
    ValueError
        If sampling frequency is not positive.
    """
    if fs <= 0:
        raise ValueError("Sampling frequency must be greater than zero.")


def valid_channel(
    signal: ArrayLike,
    min_std: float = 1e-8,
) -> bool:
    """
    Determine whether a physiological signal contains usable data.

    A channel is considered invalid when it:

    - is empty,
    - contains only NaN values,
    - contains only zeros,
    - or has near-zero variance.

    Parameters
    ----------
    signal : array-like
        Physiological signal.
    min_std : float, optional
        Minimum acceptable standard deviation.

    Returns
    -------
    bool
        True if the signal is valid, otherwise False.
    """

    x = np.asarray(signal, dtype=float)

    if x.size == 0:
        return False

    x = x[np.isfinite(x)]

    if x.size == 0:
        return False

    if np.allclose(x, 0.0):
        return False

    if np.std(x) < min_std:
        return False

    return True


# ==========================================================
# GENERIC FILTERS
# ==========================================================

def bandpass_filter(
    signal: ArrayLike,
    lowcut: float,
    highcut: float,
    fs: float = DEFAULT_FS,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.

    Parameters
    ----------
    signal : array-like
        Input signal.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float, optional
        Sampling frequency in Hz.
    order : int, optional
        Butterworth filter order.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    validate_sampling_frequency(fs)

    if not 0 < lowcut < highcut < fs / 2:
        raise ValueError(
            "Band-pass frequencies must satisfy "
            "0 < lowcut < highcut < Nyquist frequency."
        )

    x = np.asarray(signal, dtype=float)

    nyquist = fs / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="bandpass",
    )

    return filtfilt(b, a, x)


def lowpass_filter(
    signal: ArrayLike,
    cutoff: float,
    fs: float = DEFAULT_FS,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Apply a Butterworth low-pass filter.

    Parameters
    ----------
    signal : array-like
        Input signal.
    cutoff : float
        Cutoff frequency in Hz.
    fs : float, optional
        Sampling frequency in Hz.
    order : int, optional
        Butterworth filter order.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    validate_sampling_frequency(fs)

    if not 0 < cutoff < fs / 2:
        raise ValueError(
            "Low-pass cutoff must be between 0 and the Nyquist frequency."
        )

    x = np.asarray(signal, dtype=float)

    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist

    b, a = butter(
        order,
        normalized_cutoff,
        btype="lowpass",
    )

    return filtfilt(b, a, x)


def notch_filter(
    signal: ArrayLike,
    notch_freq: float = DEFAULT_NOTCH_FREQ,
    fs: float = DEFAULT_FS,
    q: float = DEFAULT_NOTCH_Q,
) -> np.ndarray:
    """
    Apply an IIR notch filter for power-line interference suppression.

    Parameters
    ----------
    signal : array-like
        Input signal.
    notch_freq : float, optional
        Notch frequency in Hz.
    fs : float, optional
        Sampling frequency in Hz.
    q : float, optional
        Quality factor.

    Returns
    -------
    numpy.ndarray
        Filtered signal.
    """

    validate_sampling_frequency(fs)

    if not 0 < notch_freq < fs / 2:
        raise ValueError(
            "Notch frequency must be between 0 and the Nyquist frequency."
        )

    if q <= 0:
        raise ValueError("Notch-filter quality factor must be positive.")

    x = np.asarray(signal, dtype=float)

    normalized_frequency = notch_freq / (fs / 2.0)

    b, a = iirnotch(
        normalized_frequency,
        q,
    )

    return filtfilt(b, a, x)


# ==========================================================
# MODALITY-SPECIFIC FILTERS
# ==========================================================

def filter_ecg(
    signal: ArrayLike,
    fs: float = DEFAULT_FS,
    lowcut: float = ECG_LOWCUT,
    highcut: float = ECG_HIGHCUT,
    notch_freq: float = DEFAULT_NOTCH_FREQ,
    notch_q: float = DEFAULT_NOTCH_Q,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Filter an ECG signal.

    Processing:
        1. Butterworth band-pass filtering.
        2. Power-line notch filtering.

    Default frequency range: 0.5–45 Hz.
    """

    filtered = bandpass_filter(
        signal=signal,
        lowcut=lowcut,
        highcut=highcut,
        fs=fs,
        order=order,
    )

    filtered = notch_filter(
        signal=filtered,
        notch_freq=notch_freq,
        fs=fs,
        q=notch_q,
    )

    return filtered


def filter_emg(
    signal: ArrayLike,
    fs: float = DEFAULT_FS,
    lowcut: float = EMG_LOWCUT,
    highcut: float = EMG_HIGHCUT,
    notch_freq: float = DEFAULT_NOTCH_FREQ,
    notch_q: float = DEFAULT_NOTCH_Q,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Filter a trapezius EMG signal.

    Processing:
        1. Butterworth band-pass filtering.
        2. Power-line notch filtering.

    Default frequency range: 20–250 Hz.
    """

    filtered = bandpass_filter(
        signal=signal,
        lowcut=lowcut,
        highcut=highcut,
        fs=fs,
        order=order,
    )

    filtered = notch_filter(
        signal=filtered,
        notch_freq=notch_freq,
        fs=fs,
        q=notch_q,
    )

    return filtered


def filter_gsr(
    signal: ArrayLike,
    fs: float = DEFAULT_FS,
    cutoff: float = GSR_CUTOFF,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """
    Filter a GSR signal using a Butterworth low-pass filter.

    Default cutoff frequency: 0.5 Hz.
    """

    return lowpass_filter(
        signal=signal,
        cutoff=cutoff,
        fs=fs,
        order=order,
    )


# ==========================================================
# DATAFRAME INTERFACE
# ==========================================================

def filter_dataframe(
    df: pd.DataFrame,
    fs: float = DEFAULT_FS,
    ecg_column: str = "ecg",
    emg_column: str = "emg_trapezius",
    gsr_column: str = "gsr",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Filter ECG, trapezius EMG, and GSR channels in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input physiological-signal DataFrame.
    fs : float, optional
        Sampling frequency in Hz.
    ecg_column : str, optional
        ECG column name.
    emg_column : str, optional
        Trapezius EMG column name.
    gsr_column : str, optional
        GSR column name.
    copy : bool, optional
        If True, operate on a copy of the input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing filtered physiological signals.

    Raises
    ------
    ValueError
        If a required channel is present but invalid.
    """

    output = df.copy() if copy else df

    channel_filters = {
        ecg_column: filter_ecg,
        emg_column: filter_emg,
        gsr_column: filter_gsr,
    }

    for column, filter_function in channel_filters.items():

        if column not in output.columns:
            continue

        signal = pd.to_numeric(
            output[column],
            errors="coerce",
        )

        if not valid_channel(signal):
            raise ValueError(
                f"Invalid or unusable physiological channel: {column}"
            )

        if signal.isna().any():
            raise ValueError(
                f"Channel '{column}' contains missing or non-numeric values."
            )

        output[column] = filter_function(
            signal.to_numpy(),
            fs=fs,
        )

    return output
