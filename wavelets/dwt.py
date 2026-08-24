"""
Modality-specific Discrete Wavelet Transform (DWT) decomposition.

This module generates multilevel DWT coefficients for ECG,
trapezius EMG, and GSR using the modality-specific mother wavelets
identified by the adaptive wavelet-selection procedure.

Canonical BSPC configuration
----------------------------
Sampling frequency : 512 Hz
Window duration    : 5.0 s
Window size        : 2560 samples
DWT extension mode : symmetric

Selected modality-specific wavelets:
    ECG            : sym4, level 6
    Trapezius EMG  : sym6, level 5
    GSR            : coif3, level 7

The functions in this module are dataset-independent and contain
no hard-coded input/output paths.

Author
------
Shreya
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Tuple, Union

import numpy as np
import pandas as pd
import pywt


ArrayLike = Union[np.ndarray, pd.Series]


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_FS = 512
DEFAULT_WINDOW_DURATION = 5.0
DEFAULT_WINDOW_SIZE = 2560

DEFAULT_DWT_MODE = "symmetric"

MINIMUM_SIGNAL_LENGTH = 32
MINIMUM_STANDARD_DEVIATION = 1e-10

DEFAULT_MAX_RECONSTRUCTION_RMSE = 1e-8


# ==========================================================
# MODALITY-SPECIFIC DWT CONFIGURATION
# ==========================================================

DEFAULT_DWT_CONFIG = {
    "ecg": {
        "wavelet": "sym4",
        "level": 6,
    },

    "emg_trapezius": {
        "wavelet": "sym6",
        "level": 5,
    },

    "gsr": {
        "wavelet": "coif3",
        "level": 7,
    },
}


# ==========================================================
# SIGNAL VALIDATION
# ==========================================================

def validate_signal(
    signal: ArrayLike,
    minimum_length: int = MINIMUM_SIGNAL_LENGTH,
    minimum_std: float = MINIMUM_STANDARD_DEVIATION,
) -> np.ndarray:
    """
    Validate a physiological signal before DWT decomposition.

    Parameters
    ----------
    signal : array-like
        Input physiological signal.
    minimum_length : int, optional
        Minimum acceptable signal length.
    minimum_std : float, optional
        Minimum acceptable standard deviation.

    Returns
    -------
    numpy.ndarray
        Validated one-dimensional floating-point signal.

    Raises
    ------
    ValueError
        If the signal is empty, non-finite, too short,
        or near-constant.
    """

    try:
        x = np.asarray(
            signal,
            dtype=np.float64,
        ).reshape(-1)

    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Signal must contain numeric values."
        ) from exc

    if x.size < minimum_length:
        raise ValueError(
            f"Signal must contain at least "
            f"{minimum_length} samples."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if np.std(x) < minimum_std:
        raise ValueError(
            "Signal is constant or near-constant."
        )

    return x


# ==========================================================
# WINDOW LENGTH VALIDATION
# ==========================================================

def validate_window_length(
    signal: ArrayLike,
    expected_samples: int = DEFAULT_WINDOW_SIZE,
) -> None:
    """
    Verify that a signal contains the expected number of samples.

    For the canonical BSPC configuration, each segment must contain
    exactly 2560 samples (5 seconds at 512 Hz).

    Parameters
    ----------
    signal : array-like
        Input signal.
    expected_samples : int, optional
        Expected segment length.

    Raises
    ------
    ValueError
        If signal length differs from the expected length.
    """

    signal_length = len(signal)

    if signal_length != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} samples, "
            f"but received {signal_length}."
        )


# ==========================================================
# COEFFICIENT LABELS
# ==========================================================

def generate_coefficient_labels(
    decomposition_level: int,
) -> List[str]:
    """
    Generate labels corresponding to ``pywt.wavedec`` output.

    Examples
    --------
    For level 5:

        A5, D5, D4, D3, D2, D1

    For level 7:

        A7, D7, D6, D5, D4, D3, D2, D1
    """

    if decomposition_level < 1:
        raise ValueError(
            "Decomposition level must be at least 1."
        )

    labels = [
        f"A{decomposition_level}"
    ]

    labels.extend(
        f"D{level}"
        for level in range(
            decomposition_level,
            0,
            -1,
        )
    )

    return labels


# ==========================================================
# VALID DWT LEVEL
# ==========================================================

def determine_valid_level(
    signal_length: int,
    wavelet_name: str,
    requested_level: int,
) -> Tuple[int, int, bool]:
    """
    Determine the mathematically valid DWT decomposition level.

    Parameters
    ----------
    signal_length : int
        Number of samples in the input signal.
    wavelet_name : str
        Mother-wavelet name.
    requested_level : int
        Desired decomposition level.

    Returns
    -------
    actual_level : int
        Decomposition level used.
    maximum_level : int
        Maximum mathematically valid level.
    level_adjusted : bool
        True if requested level had to be reduced.
    """

    if signal_length <= 0:
        raise ValueError(
            "Signal length must be greater than zero."
        )

    if requested_level < 1:
        raise ValueError(
            "Requested DWT level must be at least 1."
        )

    wavelet = pywt.Wavelet(
        wavelet_name
    )

    maximum_level = pywt.dwt_max_level(
        data_len=signal_length,
        filter_len=wavelet.dec_len,
    )

    if maximum_level < 1:
        raise ValueError(
            f"Signal is too short for DWT decomposition "
            f"using wavelet '{wavelet_name}'."
        )

    actual_level = min(
        requested_level,
        maximum_level,
    )

    level_adjusted = (
        actual_level < requested_level
    )

    return (
        int(actual_level),
        int(maximum_level),
        bool(level_adjusted),
    )


# ==========================================================
# DWT DECOMPOSITION
# ==========================================================

def perform_dwt(
    signal: ArrayLike,
    wavelet_name: str,
    requested_level: int,
    mode: str = DEFAULT_DWT_MODE,
    require_expected_length: bool = False,
    expected_samples: int = DEFAULT_WINDOW_SIZE,
) -> Dict:
    """
    Perform multilevel DWT decomposition.

    Parameters
    ----------
    signal : array-like
        Input physiological signal.
    wavelet_name : str
        Mother wavelet.
    requested_level : int
        Requested DWT decomposition level.
    mode : str, optional
        PyWavelets boundary-extension mode.
    require_expected_length : bool, optional
        If True, enforce the expected segment length.
    expected_samples : int, optional
        Expected number of samples.

    Returns
    -------
    dict
        Dictionary containing decomposition metadata and
        labeled coefficient arrays.
    """

    x = validate_signal(
        signal
    )

    if require_expected_length:
        validate_window_length(
            x,
            expected_samples=expected_samples,
        )

    (
        actual_level,
        maximum_level,
        level_adjusted,
    ) = determine_valid_level(
        signal_length=len(x),
        wavelet_name=wavelet_name,
        requested_level=requested_level,
    )

    coefficient_list = pywt.wavedec(
        data=x,
        wavelet=wavelet_name,
        mode=mode,
        level=actual_level,
    )

    labels = generate_coefficient_labels(
        actual_level
    )

    if len(coefficient_list) != len(labels):
        raise RuntimeError(
            "DWT coefficient and label counts do not match."
        )

    coefficient_dictionary = {
        label: np.asarray(
            coefficient,
            dtype=np.float64,
        )
        for label, coefficient in zip(
            labels,
            coefficient_list,
        )
    }

    return {
        "wavelet":
            wavelet_name,

        "requested_level":
            int(requested_level),

        "actual_level":
            int(actual_level),

        "maximum_level":
            int(maximum_level),

        "level_adjusted":
            bool(level_adjusted),

        "mode":
            mode,

        "signal_length":
            len(x),

        "coefficients":
            coefficient_dictionary,

        "coefficient_list":
            coefficient_list,

        "coefficient_labels":
            labels,
    }


# ==========================================================
# MODALITY-SPECIFIC DWT
# ==========================================================

def decompose_modality(
    signal: ArrayLike,
    modality: str,
    dwt_config: Mapping = DEFAULT_DWT_CONFIG,
    mode: str = DEFAULT_DWT_MODE,
    require_expected_length: bool = True,
    expected_samples: int = DEFAULT_WINDOW_SIZE,
) -> Dict:
    """
    Perform DWT using the selected wavelet for one modality.

    Parameters
    ----------
    signal : array-like
        Physiological signal.
    modality : str
        One of:
        ``ecg``,
        ``emg_trapezius``,
        ``gsr``.
    dwt_config : mapping, optional
        Modality-specific DWT configuration.
    mode : str, optional
        PyWavelets extension mode.
    require_expected_length : bool, optional
        Require canonical 2560-sample windows.
    expected_samples : int, optional
        Expected segment length.

    Returns
    -------
    dict
        DWT decomposition result.
    """

    if modality not in dwt_config:
        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Available modalities: "
            f"{list(dwt_config.keys())}"
        )

    configuration = dwt_config[
        modality
    ]

    result = perform_dwt(
        signal=signal,
        wavelet_name=configuration[
            "wavelet"
        ],
        requested_level=configuration[
            "level"
        ],
        mode=mode,
        require_expected_length=
            require_expected_length,
        expected_samples=
            expected_samples,
    )

    result["modality"] = modality

    return result


# ==========================================================
# MULTIMODAL DWT
# ==========================================================

def decompose_dataframe(
    df: pd.DataFrame,
    dwt_config: Mapping = DEFAULT_DWT_CONFIG,
    mode: str = DEFAULT_DWT_MODE,
    require_expected_length: bool = True,
    expected_samples: int = DEFAULT_WINDOW_SIZE,
) -> Dict[str, Dict]:
    """
    Decompose all configured physiological channels in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input normalized physiological segment.
    dwt_config : mapping, optional
        Modality-specific DWT configuration.
    mode : str, optional
        PyWavelets extension mode.
    require_expected_length : bool, optional
        Require each channel to contain exactly 2560 samples.
    expected_samples : int, optional
        Expected segment length.

    Returns
    -------
    dict
        Mapping from modality name to DWT decomposition result.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    results = {}

    for modality in dwt_config:

        if modality not in df.columns:
            continue

        results[modality] = decompose_modality(
            signal=df[
                modality
            ].to_numpy(),
            modality=modality,
            dwt_config=dwt_config,
            mode=mode,
            require_expected_length=
                require_expected_length,
            expected_samples=
                expected_samples,
        )

    return results


# ==========================================================
# SIGNAL RECONSTRUCTION
# ==========================================================

def reconstruct_signal(
    coefficient_list: List[np.ndarray],
    wavelet_name: str,
    original_length: int,
    mode: str = DEFAULT_DWT_MODE,
) -> np.ndarray:
    """
    Reconstruct the original signal using inverse DWT.

    Parameters
    ----------
    coefficient_list : list of numpy.ndarray
        DWT coefficients.
    wavelet_name : str
        Mother-wavelet name.
    original_length : int
        Number of samples in the original signal.
    mode : str, optional
        PyWavelets extension mode.

    Returns
    -------
    numpy.ndarray
        Reconstructed signal.
    """

    reconstructed = pywt.waverec(
        coeffs=coefficient_list,
        wavelet=wavelet_name,
        mode=mode,
    )

    reconstructed = reconstructed[
        :original_length
    ]

    return np.asarray(
        reconstructed,
        dtype=np.float64,
    )


# ==========================================================
# RECONSTRUCTION VALIDATION
# ==========================================================

def calculate_reconstruction_metrics(
    original_signal: ArrayLike,
    reconstructed_signal: ArrayLike,
    maximum_rmse: float = DEFAULT_MAX_RECONSTRUCTION_RMSE,
) -> Dict[str, Union[float, bool]]:
    """
    Calculate DWT reconstruction quality metrics.

    Metrics
    -------
    RMSE
    MAE
    Maximum absolute error
    Percentage root-mean-square difference (PRD)
    """

    original = np.asarray(
        original_signal,
        dtype=np.float64,
    )

    reconstructed = np.asarray(
        reconstructed_signal,
        dtype=np.float64,
    )

    if original.shape != reconstructed.shape:
        raise ValueError(
            "Original and reconstructed signals "
            "must have identical shapes."
        )

    error = (
        original
        - reconstructed
    )

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    maximum_absolute_error = float(
        np.max(
            np.abs(error)
        )
    )

    denominator = float(
        np.sum(
            original ** 2
        )
    )

    if denominator > 0:

        prd = float(
            100.0
            * np.sqrt(
                np.sum(
                    error ** 2
                )
                / denominator
            )
        )

    else:
        prd = np.nan

    reconstruction_valid = bool(
        np.isfinite(rmse)
        and rmse <= maximum_rmse
    )

    return {
        "reconstruction_rmse":
            rmse,

        "reconstruction_mae":
            mae,

        "maximum_absolute_error":
            maximum_absolute_error,

        "prd_percentage":
            prd,

        "reconstruction_valid":
            reconstruction_valid,
    }


# ==========================================================
# DECOMPOSITION + RECONSTRUCTION CHECK
# ==========================================================

def validate_dwt_reconstruction(
    signal: ArrayLike,
    dwt_result: Dict,
    maximum_rmse: float = DEFAULT_MAX_RECONSTRUCTION_RMSE,
) -> Dict[str, Union[float, bool]]:
    """
    Reconstruct a DWT-decomposed signal and verify numerical accuracy.
    """

    x = np.asarray(
        signal,
        dtype=np.float64,
    )

    reconstructed = reconstruct_signal(
        coefficient_list=
            dwt_result[
                "coefficient_list"
            ],
        wavelet_name=
            dwt_result[
                "wavelet"
            ],
        original_length=len(x),
        mode=dwt_result[
            "mode"
        ],
    )

    return calculate_reconstruction_metrics(
        original_signal=x,
        reconstructed_signal=
            reconstructed,
        maximum_rmse=maximum_rmse,
    )
