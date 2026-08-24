"""
Physiological signal segmentation.

This module implements fixed-length overlapping window segmentation
for multimodal physiological signals used in the pain-recognition
framework.

Canonical segmentation settings
--------------------------------
Sampling frequency : 512 Hz
Window duration    : 5.0 s
Window size        : 2560 samples
Overlap            : 50%
Step size          : 1280 samples

The segmentation functions are dataset-independent and contain
no hard-coded input/output paths.

Author
------
Shreya
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_FS = 512
DEFAULT_WINDOW_DURATION = 5.0
DEFAULT_OVERLAP = 0.50


# ==========================================================
# PARAMETER VALIDATION
# ==========================================================

def validate_segmentation_parameters(
    fs: float,
    window_duration: float,
    overlap: float,
) -> None:
    """
    Validate segmentation parameters.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    window_duration : float
        Window duration in seconds.
    overlap : float
        Fractional overlap between consecutive windows.

    Raises
    ------
    ValueError
        If any segmentation parameter is invalid.
    """

    if fs <= 0:
        raise ValueError(
            "Sampling frequency must be greater than zero."
        )

    if window_duration <= 0:
        raise ValueError(
            "Window duration must be greater than zero."
        )

    if not 0 <= overlap < 1:
        raise ValueError(
            "Overlap must satisfy 0 <= overlap < 1."
        )


# ==========================================================
# WINDOW PARAMETERS
# ==========================================================

def calculate_window_parameters(
    fs: float = DEFAULT_FS,
    window_duration: float = DEFAULT_WINDOW_DURATION,
    overlap: float = DEFAULT_OVERLAP,
) -> Tuple[int, int]:
    """
    Calculate window size and step size in samples.

    Parameters
    ----------
    fs : float, optional
        Sampling frequency in Hz.
    window_duration : float, optional
        Duration of each segment in seconds.
    overlap : float, optional
        Fractional overlap between consecutive windows.

    Returns
    -------
    tuple of int
        ``(window_size, step_size)`` in samples.

    Examples
    --------
    For the canonical BSPC configuration:

    512 Hz × 5 s = 2560 samples

    With 50% overlap:

    step size = 1280 samples
    """

    validate_segmentation_parameters(
        fs=fs,
        window_duration=window_duration,
        overlap=overlap,
    )

    window_size = int(round(fs * window_duration))

    step_size = int(
        round(window_size * (1.0 - overlap))
    )

    if window_size <= 0:
        raise ValueError(
            "Calculated window size must be greater than zero."
        )

    if step_size <= 0:
        raise ValueError(
            "Calculated step size must be greater than zero."
        )

    return window_size, step_size


# ==========================================================
# SEGMENT COUNT
# ==========================================================

def count_segments(
    n_samples: int,
    window_size: int,
    step_size: int,
) -> int:
    """
    Calculate the number of complete segments.

    Parameters
    ----------
    n_samples : int
        Number of samples in the input recording.
    window_size : int
        Number of samples in each segment.
    step_size : int
        Number of samples between consecutive segment starts.

    Returns
    -------
    int
        Number of complete segments.
    """

    if n_samples < 0:
        raise ValueError(
            "Number of samples cannot be negative."
        )

    if window_size <= 0:
        raise ValueError(
            "Window size must be greater than zero."
        )

    if step_size <= 0:
        raise ValueError(
            "Step size must be greater than zero."
        )

    if n_samples < window_size:
        return 0

    return 1 + (
        (n_samples - window_size) // step_size
    )


# ==========================================================
# DATAFRAME SEGMENTATION
# ==========================================================

def segment_dataframe(
    df: pd.DataFrame,
    fs: float = DEFAULT_FS,
    window_duration: float = DEFAULT_WINDOW_DURATION,
    overlap: float = DEFAULT_OVERLAP,
) -> List[pd.DataFrame]:
    """
    Segment a physiological-signal DataFrame into overlapping windows.

    Only complete windows are retained. Any remaining samples that
    cannot form a complete final window are discarded.

    Parameters
    ----------
    df : pandas.DataFrame
        Input physiological recording. All channels must have equal
        length and be aligned row-wise.
    fs : float, optional
        Sampling frequency in Hz.
    window_duration : float, optional
        Duration of each segment in seconds.
    overlap : float, optional
        Fractional overlap between consecutive windows.

    Returns
    -------
    list of pandas.DataFrame
        List containing segmented DataFrames.

    Notes
    -----
    For the canonical BSPC experiment:

    - fs = 512 Hz
    - window_duration = 5.0 s
    - window_size = 2560 samples
    - overlap = 0.50
    - step_size = 1280 samples
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        return []

    window_size, step_size = calculate_window_parameters(
        fs=fs,
        window_duration=window_duration,
        overlap=overlap,
    )

    total_samples = len(df)

    if total_samples < window_size:
        return []

    segments = []

    segment_id = 0

    for start_sample in range(
        0,
        total_samples - window_size + 1,
        step_size,
    ):

        end_sample = start_sample + window_size

        segment = df.iloc[
            start_sample:end_sample
        ].copy()

        # Reset the row index within each segment.
        segment.reset_index(
            drop=True,
            inplace=True,
        )

        # Store useful metadata.
        segment.attrs["segment_id"] = segment_id
        segment.attrs["start_sample"] = start_sample
        segment.attrs["end_sample"] = end_sample
        segment.attrs["window_size"] = window_size
        segment.attrs["step_size"] = step_size
        segment.attrs["sampling_frequency"] = fs
        segment.attrs["window_duration"] = window_duration
        segment.attrs["overlap"] = overlap

        segments.append(segment)

        segment_id += 1

    return segments
