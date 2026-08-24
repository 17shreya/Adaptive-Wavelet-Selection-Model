"""
Hierarchical DWT-based feature extraction.

This module extracts coefficient-level and modality-level hierarchical
features from Discrete Wavelet Transform (DWT) coefficients generated
from ECG, trapezius EMG, and GSR physiological signals.

The feature representation includes:

Statistical features
--------------------
- Mean
- Median
- Standard deviation
- Variance
- Minimum
- Maximum
- Range
- Interquartile range
- Median absolute deviation
- RMS
- Mean absolute value
- Skewness
- Kurtosis

Energy features
---------------
- Energy
- Mean energy
- Relative energy
- Log energy

Waveform features
-----------------
- Normalized waveform length
- Zero-crossing rate
- Slope-sign-change rate

Entropy features
----------------
- Shannon entropy
- Normalized Shannon entropy
- Log-energy entropy

Hjorth parameters
-----------------
- Activity
- Mobility
- Complexity

Level-aware spectral features
-----------------------------
- Effective DWT sampling frequency
- Mean frequency
- Median frequency
- Spectral entropy

Canonical BSPC configuration
----------------------------
Sampling frequency : 512 Hz
Window duration    : 5.5 s
Window size        : 2816 samples

Selected DWT configuration:
    ECG            : sym4, level 6
    Trapezius EMG  : sym6, level 5
    GSR            : coif3, level 7

This module contains no dataset paths, file traversal,
CSV saving, or experiment-specific output directories.

Author
------
Shreya
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scipy.signal import welch
from scipy.stats import kurtosis, skew


ArrayLike = Union[np.ndarray, pd.Series]


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_FS = 512

DEFAULT_WINDOW_DURATION = 5.5
DEFAULT_WINDOW_SIZE = 2816

EPSILON = 1e-12


# ==========================================================
# COEFFICIENT CLEANING
# ==========================================================

def clean_coefficient(
    coefficient: ArrayLike,
) -> np.ndarray:
    """
    Convert a coefficient array to a finite one-dimensional array.

    Parameters
    ----------
    coefficient : array-like
        Input DWT coefficient.

    Returns
    -------
    numpy.ndarray
        Finite one-dimensional coefficient array.
    """

    x = np.asarray(
        coefficient,
        dtype=np.float64,
    ).reshape(-1)

    return x[
        np.isfinite(x)
    ]


# ==========================================================
# COEFFICIENT VALIDATION
# ==========================================================

def validate_coefficient(
    coefficient: ArrayLike,
    minimum_length: int = 1,
) -> np.ndarray:
    """
    Validate and clean a DWT coefficient.

    A small number of NaN/Inf values are linearly interpolated.

    Parameters
    ----------
    coefficient : array-like
        Input DWT coefficient.
    minimum_length : int, optional
        Minimum acceptable coefficient length.

    Returns
    -------
    numpy.ndarray
        Validated coefficient.

    Raises
    ------
    ValueError
        If coefficient is invalid.
    """

    try:

        x = np.asarray(
            coefficient,
            dtype=np.float64,
        ).reshape(-1)

    except (TypeError, ValueError) as exc:

        raise ValueError(
            "Coefficient must contain numeric values."
        ) from exc

    if len(x) < minimum_length:

        raise ValueError(
            f"Coefficient must contain at least "
            f"{minimum_length} samples."
        )

    if not np.all(
        np.isfinite(x)
    ):

        x = (
            pd.Series(x)
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .interpolate(
                method="linear",
                limit_direction="both",
            )
            .to_numpy(
                dtype=np.float64
            )
        )

    if not np.all(
        np.isfinite(x)
    ):

        raise ValueError(
            "Coefficient contains unresolved "
            "non-finite values."
        )

    return x


# ==========================================================
# HJORTH PARAMETERS
# ==========================================================

def calculate_hjorth_parameters(
    coefficient: ArrayLike,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Calculate Hjorth activity, mobility and complexity.
    """

    x = clean_coefficient(
        coefficient
    )

    if len(x) < 3:

        return {
            "hjorth_activity": 0.0,
            "hjorth_mobility": 0.0,
            "hjorth_complexity": 0.0,
        }

    first_difference = np.diff(
        x
    )

    second_difference = np.diff(
        first_difference
    )

    activity = float(
        np.var(x)
    )

    first_variance = float(
        np.var(
            first_difference
        )
    )

    second_variance = float(
        np.var(
            second_difference
        )
    )

    if activity <= epsilon:

        mobility = 0.0

    else:

        mobility = float(
            np.sqrt(
                first_variance
                / activity
            )
        )

    if (
        first_variance <= epsilon
        or mobility <= epsilon
    ):

        complexity = 0.0

    else:

        derivative_mobility = np.sqrt(
            second_variance
            / first_variance
        )

        complexity = float(
            derivative_mobility
            / mobility
        )

    return {
        "hjorth_activity":
            activity,

        "hjorth_mobility":
            mobility,

        "hjorth_complexity":
            complexity,
    }


# ==========================================================
# ZERO-CROSSING RATE
# ==========================================================

def calculate_zero_crossing_rate(
    coefficient: ArrayLike,
    threshold: Optional[float] = None,
) -> float:
    """
    Calculate thresholded normalized zero-crossing rate.
    """

    x = clean_coefficient(
        coefficient
    )

    if len(x) < 2:

        return 0.0

    if threshold is None:

        threshold = (
            0.01
            * np.std(x)
        )

    left_values = x[:-1]
    right_values = x[1:]

    sign_change = (
        left_values
        * right_values
        < 0
    )

    sufficient_difference = (
        np.abs(
            left_values
            - right_values
        )
        >= threshold
    )

    crossing_count = int(
        np.sum(
            sign_change
            & sufficient_difference
        )
    )

    return float(
        crossing_count
        / max(
            len(x) - 1,
            1,
        )
    )


# ==========================================================
# SLOPE-SIGN-CHANGE RATE
# ==========================================================

def calculate_slope_sign_change_rate(
    coefficient: ArrayLike,
    threshold: Optional[float] = None,
) -> float:
    """
    Calculate thresholded normalized slope-sign-change rate.
    """

    x = clean_coefficient(
        coefficient
    )

    if len(x) < 3:

        return 0.0

    if threshold is None:

        threshold = (
            0.01
            * np.std(x)
        )

    left_slope = (
        x[1:-1]
        - x[:-2]
    )

    right_slope = (
        x[1:-1]
        - x[2:]
    )

    sign_change = (
        left_slope
        * right_slope
        > 0
    )

    sufficient_change = (
        (np.abs(left_slope) >= threshold)
        |
        (np.abs(right_slope) >= threshold)
    )

    change_count = int(
        np.sum(
            sign_change
            & sufficient_change
        )
    )

    return float(
        change_count
        / max(
            len(x) - 2,
            1,
        )
    )


# ==========================================================
# NORMALIZED WAVEFORM LENGTH
# ==========================================================

def calculate_normalized_waveform_length(
    coefficient: ArrayLike,
) -> float:
    """
    Calculate waveform length normalized by coefficient size.
    """

    x = clean_coefficient(
        coefficient
    )

    if len(x) < 2:

        return 0.0

    waveform_length = float(
        np.sum(
            np.abs(
                np.diff(x)
            )
        )
    )

    return float(
        waveform_length
        / max(
            len(x) - 1,
            1,
        )
    )


# ==========================================================
# ENERGY-BASED ENTROPY
# ==========================================================

def calculate_energy_entropy_features(
    coefficient: ArrayLike,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Calculate energy-based entropy features.

    Returns
    -------
    dict
        Shannon entropy,
        normalized Shannon entropy,
        and log-energy entropy.
    """

    x = clean_coefficient(
        coefficient
    )

    if len(x) == 0:

        return {
            "shannon_entropy": 0.0,
            "normalized_shannon_entropy": 0.0,
            "log_energy_entropy": 0.0,
        }

    squared_values = (
        x ** 2
    )

    total_energy = float(
        np.sum(
            squared_values
        )
    )

    if total_energy <= epsilon:

        return {
            "shannon_entropy": 0.0,
            "normalized_shannon_entropy": 0.0,
            "log_energy_entropy": 0.0,
        }

    probabilities = (
        squared_values
        / total_energy
    )

    probabilities = probabilities[
        probabilities > epsilon
    ]

    shannon_entropy = float(
        -np.sum(
            probabilities
            * np.log2(
                probabilities
            )
        )
    )

    maximum_entropy = float(
        np.log2(
            max(
                len(x),
                2,
            )
        )
    )

    normalized_entropy = float(
        shannon_entropy
        / max(
            maximum_entropy,
            epsilon,
        )
    )

    log_energy_entropy = float(
        np.sum(
            np.log(
                squared_values
                + epsilon
            )
        )
    )

    return {
        "shannon_entropy":
            shannon_entropy,

        "normalized_shannon_entropy":
            normalized_entropy,

        "log_energy_entropy":
            log_energy_entropy,
    }


# ==========================================================
# EFFECTIVE DWT SAMPLING FREQUENCY
# ==========================================================

def calculate_effective_sampling_frequency(
    original_sampling_rate: float,
    coefficient_level: int,
) -> float:
    """
    Approximate the effective DWT coefficient sampling frequency.

    fs_effective = fs_original / (2 ** level)
    """

    coefficient_level = int(
        coefficient_level
    )

    if coefficient_level < 1:

        raise ValueError(
            "Coefficient level must be at least 1."
        )

    if original_sampling_rate <= 0:

        raise ValueError(
            "Sampling frequency must be positive."
        )

    return float(
        original_sampling_rate
        / (
            2 ** coefficient_level
        )
    )


# ==========================================================
# LEVEL-AWARE SPECTRAL FEATURES
# ==========================================================

def calculate_frequency_features(
    coefficient: ArrayLike,
    effective_sampling_frequency: float,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Calculate coefficient-level spectral descriptors using
    Welch power spectral density.
    """

    x = clean_coefficient(
        coefficient
    )

    if (
        len(x) < 8
        or effective_sampling_frequency <= 0
    ):

        return {
            "mean_frequency": 0.0,
            "median_frequency": 0.0,
            "spectral_entropy": 0.0,
        }

    frequencies, power_spectrum = welch(
        x,
        fs=effective_sampling_frequency,
        nperseg=min(
            256,
            len(x),
        ),
    )

    total_power = float(
        np.sum(
            power_spectrum
        )
    )

    if total_power <= epsilon:

        return {
            "mean_frequency": 0.0,
            "median_frequency": 0.0,
            "spectral_entropy": 0.0,
        }

    mean_frequency = float(
        np.sum(
            frequencies
            * power_spectrum
        )
        / total_power
    )

    cumulative_power = np.cumsum(
        power_spectrum
    )

    median_index = int(
        np.searchsorted(
            cumulative_power,
            cumulative_power[-1]
            / 2.0,
        )
    )

    median_index = min(
        median_index,
        len(frequencies) - 1,
    )

    median_frequency = float(
        frequencies[
            median_index
        ]
    )

    power_probabilities = (
        power_spectrum
        / total_power
    )

    power_probabilities = (
        power_probabilities[
            power_probabilities
            > epsilon
        ]
    )

    spectral_entropy = float(
        -np.sum(
            power_probabilities
            * np.log2(
                power_probabilities
            )
        )
    )

    spectral_entropy = float(
        spectral_entropy
        / np.log2(
            max(
                len(power_spectrum),
                2,
            )
        )
    )

    return {
        "mean_frequency":
            mean_frequency,

        "median_frequency":
            median_frequency,

        "spectral_entropy":
            spectral_entropy,
    }


# ==========================================================
# COMPLETE COEFFICIENT FEATURE EXTRACTION
# ==========================================================

def extract_coefficient_features(
    coefficient: ArrayLike,
    coefficient_level: int,
    original_sampling_rate: float = DEFAULT_FS,
    total_modality_energy: Optional[float] = None,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Extract the complete hierarchical feature set from
    one DWT coefficient.

    Parameters
    ----------
    coefficient : array-like
        DWT approximation or detail coefficient.
    coefficient_level : int
        DWT level associated with the coefficient.
    original_sampling_rate : float, optional
        Original physiological signal sampling frequency.
    total_modality_energy : float, optional
        Sum of energy across all coefficients for the same
        modality and segment.

    Returns
    -------
    dict
        Extracted feature values.
    """

    x = validate_coefficient(
        coefficient
    )

    mean_value = float(
        np.mean(x)
    )

    median_value = float(
        np.median(x)
    )

    standard_deviation = float(
        np.std(x)
    )

    variance = float(
        np.var(x)
    )

    minimum_value = float(
        np.min(x)
    )

    maximum_value = float(
        np.max(x)
    )

    coefficient_range = float(
        maximum_value
        - minimum_value
    )

    first_quartile, third_quartile = (
        np.percentile(
            x,
            [25, 75],
        )
    )

    interquartile_range = float(
        third_quartile
        - first_quartile
    )

    median_absolute_deviation = float(
        np.median(
            np.abs(
                x
                - median_value
            )
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                x ** 2
            )
        )
    )

    mean_absolute_value = float(
        np.mean(
            np.abs(x)
        )
    )

    energy = float(
        np.sum(
            x ** 2
        )
    )

    mean_energy = float(
        energy
        / len(x)
    )

    if (
        total_modality_energy is not None
        and total_modality_energy > epsilon
    ):

        relative_energy = float(
            energy
            / total_modality_energy
        )

    else:

        relative_energy = np.nan

    log_energy = float(
        np.log(
            energy
            + epsilon
        )
    )

    if standard_deviation > epsilon:

        skewness_value = float(
            skew(
                x,
                bias=False,
            )
        )

        kurtosis_value = float(
            kurtosis(
                x,
                fisher=True,
                bias=False,
            )
        )

    else:

        skewness_value = 0.0
        kurtosis_value = 0.0

    hjorth_features = (
        calculate_hjorth_parameters(
            x,
            epsilon=epsilon,
        )
    )

    entropy_features = (
        calculate_energy_entropy_features(
            x,
            epsilon=epsilon,
        )
    )

    effective_sampling_frequency = (
        calculate_effective_sampling_frequency(
            original_sampling_rate=
                original_sampling_rate,
            coefficient_level=
                coefficient_level,
        )
    )

    frequency_features = (
        calculate_frequency_features(
            coefficient=x,
            effective_sampling_frequency=
                effective_sampling_frequency,
            epsilon=epsilon,
        )
    )

    return {
        "mean":
            mean_value,

        "median":
            median_value,

        "standard_deviation":
            standard_deviation,

        "variance":
            variance,

        "minimum":
            minimum_value,

        "maximum":
            maximum_value,

        "range":
            coefficient_range,

        "interquartile_range":
            interquartile_range,

        "median_absolute_deviation":
            median_absolute_deviation,

        "rms":
            rms,

        "mean_absolute_value":
            mean_absolute_value,

        "energy":
            energy,

        "mean_energy":
            mean_energy,

        "relative_energy":
            relative_energy,

        "log_energy":
            log_energy,

        "skewness":
            skewness_value,

        "kurtosis":
            kurtosis_value,

        "normalized_waveform_length":
            calculate_normalized_waveform_length(
                x
            ),

        "zero_crossing_rate":
            calculate_zero_crossing_rate(
                x
            ),

        "slope_sign_change_rate":
            calculate_slope_sign_change_rate(
                x
            ),

        "effective_sampling_frequency":
            effective_sampling_frequency,

        **entropy_features,

        **hjorth_features,

        **frequency_features,
    }


# ==========================================================
# COEFFICIENT LABEL PARSING
# ==========================================================

def parse_coefficient_label(
    coefficient_label: str,
) -> Tuple[str, int]:
    """
    Parse DWT coefficient labels such as A6 or D4.

    Returns
    -------
    coefficient_type : str
        'A' or 'D'.
    coefficient_level : int
        DWT decomposition level.
    """

    match = re.fullmatch(
        r"([AD])(\d+)",
        str(
            coefficient_label
        ).strip(),
        flags=re.IGNORECASE,
    )

    if match is None:

        raise ValueError(
            f"Invalid coefficient label: "
            f"{coefficient_label}"
        )

    coefficient_type = (
        match.group(1).upper()
    )

    coefficient_level = int(
        match.group(2)
    )

    return (
        coefficient_type,
        coefficient_level,
    )


# ==========================================================
# COEFFICIENT SORTING
# ==========================================================

def coefficient_sort_key(
    coefficient_label: str,
) -> Tuple[int, int]:
    """
    Sort approximation first followed by descending details.

    Example
    -------
    A6, D6, D5, D4, D3, D2, D1
    """

    coefficient_type, level = (
        parse_coefficient_label(
            coefficient_label
        )
    )

    if coefficient_type == "A":

        return (
            0,
            -level,
        )

    return (
        1,
        -level,
    )


# ==========================================================
# TOTAL MODALITY ENERGY
# ==========================================================

def calculate_total_modality_energy(
    coefficients: Mapping[str, ArrayLike],
) -> float:
    """
    Calculate total energy across all DWT coefficients
    belonging to one modality and one segment.
    """

    total_energy = 0.0

    for coefficient in coefficients.values():

        x = validate_coefficient(
            coefficient
        )

        total_energy += float(
            np.sum(
                x ** 2
            )
        )

    return float(
        total_energy
    )


# ==========================================================
# ONE MODALITY HIERARCHICAL FEATURES
# ==========================================================

def extract_modality_hierarchical_features(
    dwt_result: Mapping,
    modality: Optional[str] = None,
    original_sampling_rate: float = DEFAULT_FS,
) -> Tuple[Dict, List[Dict]]:
    """
    Extract hierarchical features for one modality and segment.

    This function directly accepts the dictionary produced by
    ``wavelets.dwt.perform_dwt`` or ``decompose_modality``.

    Parameters
    ----------
    dwt_result : mapping
        DWT decomposition result containing a ``coefficients``
        dictionary.
    modality : str, optional
        Physiological modality name.
    original_sampling_rate : float, optional
        Original signal sampling frequency.

    Returns
    -------
    wide_record : dict
        One wide hierarchical feature record.
    coefficient_records : list of dict
        Long-form coefficient-level feature records.
    """

    if "coefficients" not in dwt_result:

        raise ValueError(
            "DWT result must contain a "
            "'coefficients' dictionary."
        )

    coefficients = dwt_result[
        "coefficients"
    ]

    if not isinstance(
        coefficients,
        Mapping,
    ):

        raise TypeError(
            "'coefficients' must be a mapping."
        )

    if not coefficients:

        raise ValueError(
            "No DWT coefficients were provided."
        )

    if modality is None:

        modality = dwt_result.get(
            "modality"
        )

    if modality is None:

        raise ValueError(
            "Modality must be provided."
        )

    total_modality_energy = (
        calculate_total_modality_energy(
            coefficients
        )
    )

    ordered_labels = sorted(
        coefficients.keys(),
        key=coefficient_sort_key,
    )

    # ------------------------------------------------------
    # Wide hierarchical representation
    # ------------------------------------------------------

    wide_record = {
        "modality":
            modality,

        "wavelet":
            dwt_result.get(
                "wavelet"
            ),

        "decomposition_level":
            dwt_result.get(
                "actual_level"
            ),

        "number_of_coefficients":
            len(coefficients),

        "total_modality_energy":
            total_modality_energy,
    }

    # ------------------------------------------------------
    # Long coefficient-level representation
    # ------------------------------------------------------

    coefficient_records = []

    for label in ordered_labels:

        coefficient = coefficients[
            label
        ]

        (
            coefficient_type,
            coefficient_level,
        ) = parse_coefficient_label(
            label
        )

        features = (
            extract_coefficient_features(
                coefficient=coefficient,
                coefficient_level=
                    coefficient_level,
                original_sampling_rate=
                    original_sampling_rate,
                total_modality_energy=
                    total_modality_energy,
            )
        )

        coefficient_record = {
            "modality":
                modality,

            "wavelet":
                dwt_result.get(
                    "wavelet"
                ),

            "coefficient":
                label,

            "coefficient_type":
                coefficient_type,

            "coefficient_level":
                coefficient_level,

            "coefficient_length":
                len(
                    coefficient
                ),

            "total_modality_energy":
                total_modality_energy,

            **features,
        }

        coefficient_records.append(
            coefficient_record
        )

        for (
            feature_name,
            feature_value,
        ) in features.items():

            column_name = (
                f"{modality}_"
                f"{label}_"
                f"{feature_name}"
            )

            wide_record[
                column_name
            ] = feature_value

    return (
        wide_record,
        coefficient_records,
    )


# ==========================================================
# MULTIMODAL HIERARCHICAL FEATURE EXTRACTION
# ==========================================================

def extract_hierarchical_features(
    dwt_results: Mapping[str, Mapping],
    metadata: Optional[Mapping] = None,
    original_sampling_rate: float = DEFAULT_FS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract hierarchical features from all available modalities.

    Parameters
    ----------
    dwt_results : mapping
        Dictionary containing modality-specific DWT results.
        Expected modalities are:
        ``ecg``, ``emg_trapezius``, and ``gsr``.
    metadata : mapping, optional
        Sample metadata such as subject ID, pain class,
        trial ID, and segment ID.
    original_sampling_rate : float, optional
        Original physiological signal sampling rate.

    Returns
    -------
    hierarchical_dataframe : pandas.DataFrame
        One hierarchical row per modality.
    coefficient_dataframe : pandas.DataFrame
        Long-form coefficient-level feature representation.
    """

    hierarchical_records = []
    coefficient_records = []

    metadata = (
        dict(metadata)
        if metadata is not None
        else {}
    )

    for modality, dwt_result in (
        dwt_results.items()
    ):

        (
            wide_record,
            long_records,
        ) = (
            extract_modality_hierarchical_features(
                dwt_result=dwt_result,
                modality=modality,
                original_sampling_rate=
                    original_sampling_rate,
            )
        )

        wide_record = {
            **metadata,
            **wide_record,
        }

        hierarchical_records.append(
            wide_record
        )

        for record in long_records:

            coefficient_records.append(
                {
                    **metadata,
                    **record,
                }
            )

    hierarchical_dataframe = (
        pd.DataFrame(
            hierarchical_records
        )
    )

    coefficient_dataframe = (
        pd.DataFrame(
            coefficient_records
        )
    )

    return (
        hierarchical_dataframe,
        coefficient_dataframe,
    )
