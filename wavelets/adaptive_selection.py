"""
Adaptive modality-specific mother-wavelet selection.

This module implements the adaptive wavelet-selection procedure used
in the multimodal physiological pain-recognition framework.

For each physiological modality, candidate mother wavelets are
evaluated using:

    1. Normalized reconstruction error
    2. Energy preservation
    3. Normalized Shannon entropy
    4. Coefficient sparsity
    5. Compression ratio

Candidate criteria are normalized within each modality. Pareto-front
membership is identified and an equal-weight multi-criteria score is
used to rank the candidate wavelets.

Canonical BSPC configuration
----------------------------
Sampling frequency : 512 Hz
Window duration    : 5.0 s
Window size        : 2560 samples

Candidate decomposition levels:
    ECG            : Level 6
    Trapezius EMG  : Level 5
    GSR            : Level 7

Final wavelets identified by the reported experiment:
    ECG            : sym4, Level 6
    Trapezius EMG  : sym6, Level 5
    GSR            : coif3, Level 7

The functions in this module are dataset-independent and contain
no hard-coded input/output paths.

Author
------
Shreya
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Tuple, Union

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
MAXIMUM_NONFINITE_RATIO = 0.05

EPSILON = 1e-12


# ==========================================================
# CANDIDATE MOTHER-WAVELET SPACE
# ==========================================================

DEFAULT_WAVELET_CONFIG = {
    "ecg": {
        "candidate_wavelets": (
            "db2",
            "db4",
            "db6",
            "sym4",
            "sym5",
            "coif3",
        ),
        "requested_level": 6,
    },

    "emg_trapezius": {
        "candidate_wavelets": (
            "db4",
            "db6",
            "db8",
            "sym6",
            "sym8",
            "coif3",
            "coif5",
        ),
        "requested_level": 5,
    },

    "gsr": {
        "candidate_wavelets": (
            "db2",
            "db4",
            "sym4",
            "sym5",
            "coif3",
        ),
        "requested_level": 7,
    },
}


# ==========================================================
# REPORTED FINAL SELECTION
# ==========================================================

REPORTED_SELECTED_WAVELETS = {
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
    maximum_nonfinite_ratio: float = MAXIMUM_NONFINITE_RATIO,
) -> np.ndarray:
    """
    Validate and clean one physiological signal.

    Signals containing a small number of NaN or infinite values are
    linearly interpolated. Signals with excessive missing values,
    insufficient length, or near-zero variance are rejected.

    Parameters
    ----------
    signal : array-like
        Input physiological signal.
    minimum_length : int, optional
        Minimum acceptable number of samples.
    minimum_std : float, optional
        Minimum acceptable signal standard deviation.
    maximum_nonfinite_ratio : float, optional
        Maximum proportion of NaN/Inf values allowed.

    Returns
    -------
    numpy.ndarray
        Validated floating-point signal.

    Raises
    ------
    ValueError
        If the signal is invalid.
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
            f"Signal must contain at least {minimum_length} samples."
        )

    nonfinite_mask = ~np.isfinite(x)

    nonfinite_ratio = (
        np.count_nonzero(nonfinite_mask) / x.size
    )

    if nonfinite_ratio > maximum_nonfinite_ratio:
        raise ValueError(
            "Signal contains excessive NaN or infinite values."
        )

    if np.any(nonfinite_mask):
        x = (
            pd.Series(x)
            .replace([np.inf, -np.inf], np.nan)
            .interpolate(
                method="linear",
                limit_direction="both",
            )
            .to_numpy(dtype=np.float64)
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Signal interpolation failed."
        )

    if np.std(x) < minimum_std:
        raise ValueError(
            "Signal is constant or near-constant."
        )

    return x


# ==========================================================
# DWT LEVEL VALIDATION
# ==========================================================

def determine_common_level(
    signal_length: int,
    candidate_wavelets: Iterable[str],
    requested_level: int,
) -> Tuple[int, Dict[str, int]]:
    """
    Determine a common valid decomposition level.

    The same decomposition level is used for all candidate wavelets
    within a modality to ensure fair comparison.

    Parameters
    ----------
    signal_length : int
        Number of signal samples.
    candidate_wavelets : iterable of str
        Candidate mother-wavelet names.
    requested_level : int
        Desired decomposition level.

    Returns
    -------
    common_level : int
        Highest common valid decomposition level not exceeding
        the requested level.
    maximum_levels : dict
        Maximum valid DWT level for each candidate wavelet.
    """

    if signal_length <= 0:
        raise ValueError(
            "Signal length must be greater than zero."
        )

    if requested_level < 1:
        raise ValueError(
            "Requested decomposition level must be at least 1."
        )

    candidate_wavelets = tuple(candidate_wavelets)

    if not candidate_wavelets:
        raise ValueError(
            "At least one candidate wavelet is required."
        )

    maximum_levels = {}

    for wavelet_name in candidate_wavelets:

        wavelet = pywt.Wavelet(
            wavelet_name
        )

        maximum_level = pywt.dwt_max_level(
            data_len=signal_length,
            filter_len=wavelet.dec_len,
        )

        maximum_levels[wavelet_name] = int(
            maximum_level
        )

    minimum_supported_level = min(
        maximum_levels.values()
    )

    common_level = min(
        requested_level,
        minimum_supported_level,
    )

    if common_level < 1:
        raise ValueError(
            "No common valid DWT level exists for "
            "the candidate wavelet set."
        )

    return int(common_level), maximum_levels


# ==========================================================
# NOISE ESTIMATION
# ==========================================================

def estimate_noise_sigma(
    detail_coefficient: ArrayLike,
) -> float:
    """
    Estimate noise standard deviation using MAD.

    The median absolute deviation of the finest-scale detail
    coefficient is used.

    sigma = MAD / 0.6745
    """

    detail = np.asarray(
        detail_coefficient,
        dtype=np.float64,
    )

    median = np.median(detail)

    mad = np.median(
        np.abs(detail - median)
    )

    return float(
        mad / 0.6745
    )


# ==========================================================
# UNIVERSAL THRESHOLD
# ==========================================================

def calculate_universal_threshold(
    sigma: float,
    signal_length: int,
) -> float:
    """
    Calculate the universal wavelet threshold.

    tau = sigma * sqrt(2 * log(N))
    """

    if signal_length <= 0:
        raise ValueError(
            "Signal length must be greater than zero."
        )

    threshold = sigma * np.sqrt(
        2.0 * np.log(
            max(signal_length, 2)
        )
    )

    return float(threshold)


# ==========================================================
# COEFFICIENT THRESHOLDING
# ==========================================================

def threshold_coefficients(
    coefficients: List[np.ndarray],
    signal_length: int,
) -> Tuple[List[np.ndarray], float, float]:
    """
    Apply soft thresholding to DWT detail coefficients.

    The approximation coefficient is preserved unchanged.

    Parameters
    ----------
    coefficients : list of numpy.ndarray
        DWT coefficients returned by ``pywt.wavedec``.
    signal_length : int
        Original signal length.

    Returns
    -------
    thresholded_coefficients : list
        Thresholded DWT coefficients.
    threshold : float
        Universal threshold.
    sigma : float
        Estimated noise standard deviation.
    """

    if len(coefficients) < 2:
        raise ValueError(
            "DWT must contain approximation and detail coefficients."
        )

    finest_detail = coefficients[-1]

    sigma = estimate_noise_sigma(
        finest_detail
    )

    threshold = calculate_universal_threshold(
        sigma=sigma,
        signal_length=signal_length,
    )

    thresholded = [
        np.asarray(
            coefficients[0],
            dtype=np.float64,
        ).copy()
    ]

    for detail in coefficients[1:]:

        thresholded_detail = pywt.threshold(
            data=detail,
            value=threshold,
            mode="soft",
        )

        thresholded.append(
            np.asarray(
                thresholded_detail,
                dtype=np.float64,
            )
        )

    return thresholded, threshold, sigma


# ==========================================================
# SIGNAL RECONSTRUCTION
# ==========================================================

def reconstruct_signal(
    coefficients: List[np.ndarray],
    wavelet_name: str,
    original_length: int,
    mode: str = DEFAULT_DWT_MODE,
) -> np.ndarray:
    """
    Reconstruct a signal from DWT coefficients.
    """

    reconstructed = pywt.waverec(
        coeffs=coefficients,
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
# NORMALIZED SHANNON ENTROPY
# ==========================================================

def calculate_normalized_entropy(
    coefficients: List[np.ndarray],
    epsilon: float = EPSILON,
) -> float:
    """
    Calculate normalized Shannon entropy of wavelet coefficients.

    Lower entropy indicates a more compact representation.
    """

    flattened = np.concatenate(
        [
            np.asarray(
                coefficient,
                dtype=np.float64,
            ).reshape(-1)
            for coefficient in coefficients
        ]
    )

    energy = flattened ** 2

    total_energy = np.sum(
        energy
    )

    if total_energy <= epsilon:
        return 0.0

    probabilities = (
        energy / total_energy
    )

    probabilities = probabilities[
        probabilities > epsilon
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    maximum_entropy = np.log2(
        max(len(flattened), 2)
    )

    return float(
        entropy / maximum_entropy
    )


# ==========================================================
# SPARSITY METRICS
# ==========================================================

def calculate_sparsity_metrics(
    original_coefficients: List[np.ndarray],
    thresholded_coefficients: List[np.ndarray],
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Calculate coefficient sparsity and compression ratio.
    """

    original_flattened = np.concatenate(
        [
            np.asarray(coefficient).reshape(-1)
            for coefficient in original_coefficients
        ]
    )

    thresholded_flattened = np.concatenate(
        [
            np.asarray(coefficient).reshape(-1)
            for coefficient in thresholded_coefficients
        ]
    )

    total_coefficients = len(
        original_flattened
    )

    retained_coefficients = int(
        np.count_nonzero(
            np.abs(thresholded_flattened)
            > epsilon
        )
    )

    zero_coefficients = (
        total_coefficients
        - retained_coefficients
    )

    sparsity = (
        zero_coefficients
        / total_coefficients
    )

    compression_ratio = (
        total_coefficients
        / max(retained_coefficients, 1)
    )

    return {
        "total_coefficients":
            int(total_coefficients),

        "retained_coefficients":
            int(retained_coefficients),

        "zero_coefficients":
            int(zero_coefficients),

        "sparsity":
            float(sparsity),

        "compression_ratio":
            float(compression_ratio),
    }


# ==========================================================
# RECONSTRUCTION METRICS
# ==========================================================

def calculate_reconstruction_metrics(
    original_signal: ArrayLike,
    reconstructed_signal: ArrayLike,
    epsilon: float = EPSILON,
) -> Dict[str, float]:
    """
    Calculate reconstruction-error and energy metrics.
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
            "Original and reconstructed signals must "
            "have identical shapes."
        )

    error = original - reconstructed

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    signal_range = float(
        np.max(original)
        - np.min(original)
    )

    normalized_rmse = (
        rmse
        / max(signal_range, epsilon)
    )

    original_energy = float(
        np.sum(
            original ** 2
        )
    )

    reconstructed_energy = float(
        np.sum(
            reconstructed ** 2
        )
    )

    energy_preservation = (
        reconstructed_energy
        / max(original_energy, epsilon)
    )

    energy_loss = abs(
        1.0 - energy_preservation
    )

    return {
        "reconstruction_rmse":
            rmse,

        "normalized_rmse":
            float(normalized_rmse),

        "original_energy":
            original_energy,

        "reconstructed_energy":
            reconstructed_energy,

        "energy_preservation":
            float(energy_preservation),

        "energy_loss":
            float(energy_loss),
    }


# ==========================================================
# EVALUATE ONE CANDIDATE WAVELET
# ==========================================================

def evaluate_candidate_wavelet(
    signal: ArrayLike,
    wavelet_name: str,
    decomposition_level: int,
    mode: str = DEFAULT_DWT_MODE,
) -> Dict[str, float]:
    """
    Evaluate one candidate mother wavelet on one signal.

    Parameters
    ----------
    signal : array-like
        Physiological signal.
    wavelet_name : str
        Candidate mother wavelet.
    decomposition_level : int
        DWT decomposition level.
    mode : str, optional
        PyWavelets signal-extension mode.

    Returns
    -------
    dict
        Candidate-wavelet quality metrics.
    """

    x = validate_signal(
        signal
    )

    coefficients = pywt.wavedec(
        data=x,
        wavelet=wavelet_name,
        mode=mode,
        level=decomposition_level,
    )

    (
        thresholded_coefficients,
        threshold,
        estimated_sigma,
    ) = threshold_coefficients(
        coefficients=coefficients,
        signal_length=len(x),
    )

    reconstructed_signal = reconstruct_signal(
        coefficients=thresholded_coefficients,
        wavelet_name=wavelet_name,
        original_length=len(x),
        mode=mode,
    )

    reconstruction_metrics = (
        calculate_reconstruction_metrics(
            original_signal=x,
            reconstructed_signal=reconstructed_signal,
        )
    )

    sparsity_metrics = (
        calculate_sparsity_metrics(
            original_coefficients=coefficients,
            thresholded_coefficients=
                thresholded_coefficients,
        )
    )

    normalized_entropy = (
        calculate_normalized_entropy(
            thresholded_coefficients
        )
    )

    return {
        "threshold":
            float(threshold),

        "estimated_noise_sigma":
            float(estimated_sigma),

        "normalized_entropy":
            float(normalized_entropy),

        **reconstruction_metrics,
        **sparsity_metrics,
    }


# ==========================================================
# EVALUATE ALL CANDIDATES FOR ONE SIGNAL
# ==========================================================

def evaluate_wavelet_candidates(
    signal: ArrayLike,
    modality: str,
    wavelet_config: Mapping = DEFAULT_WAVELET_CONFIG,
    mode: str = DEFAULT_DWT_MODE,
) -> pd.DataFrame:
    """
    Evaluate all candidate wavelets for a single signal.

    Parameters
    ----------
    signal : array-like
        Physiological signal.
    modality : str
        Modality name: ``ecg``, ``emg_trapezius``, or ``gsr``.
    wavelet_config : mapping, optional
        Candidate-wavelet configuration.
    mode : str, optional
        PyWavelets extension mode.

    Returns
    -------
    pandas.DataFrame
        Candidate-level evaluation metrics.
    """

    if modality not in wavelet_config:
        raise ValueError(
            f"Unknown modality: {modality}"
        )

    x = validate_signal(
        signal
    )

    configuration = wavelet_config[
        modality
    ]

    candidate_wavelets = configuration[
        "candidate_wavelets"
    ]

    requested_level = configuration[
        "requested_level"
    ]

    common_level, maximum_levels = (
        determine_common_level(
            signal_length=len(x),
            candidate_wavelets=
                candidate_wavelets,
            requested_level=
                requested_level,
        )
    )

    records = []

    for wavelet_name in candidate_wavelets:

        metrics = evaluate_candidate_wavelet(
            signal=x,
            wavelet_name=wavelet_name,
            decomposition_level=common_level,
            mode=mode,
        )

        records.append(
            {
                "modality":
                    modality,

                "wavelet":
                    wavelet_name,

                "requested_level":
                    requested_level,

                "common_level":
                    common_level,

                "wavelet_maximum_level":
                    maximum_levels[
                        wavelet_name
                    ],

                "signal_length":
                    len(x),

                **metrics,
            }
        )

    return pd.DataFrame(
        records
    )


# ==========================================================
# AGGREGATE SEGMENT-LEVEL METRICS
# ==========================================================

def aggregate_wavelet_metrics(
    segment_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate segment-level wavelet metrics by modality and wavelet.
    """

    required_columns = {
        "modality",
        "wavelet",
        "common_level",
        "normalized_rmse",
        "energy_preservation",
        "energy_loss",
        "normalized_entropy",
        "sparsity",
        "compression_ratio",
    }

    missing = (
        required_columns
        - set(segment_metrics.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    aggregated = (
        segment_metrics
        .groupby(
            [
                "modality",
                "wavelet",
            ],
            as_index=False,
        )
        .agg(
            number_of_segments=(
                "wavelet",
                "size",
            ),

            decomposition_level=(
                "common_level",
                "median",
            ),

            mean_normalized_rmse=(
                "normalized_rmse",
                "mean",
            ),

            std_normalized_rmse=(
                "normalized_rmse",
                "std",
            ),

            mean_energy_preservation=(
                "energy_preservation",
                "mean",
            ),

            mean_energy_loss=(
                "energy_loss",
                "mean",
            ),

            mean_normalized_entropy=(
                "normalized_entropy",
                "mean",
            ),

            mean_sparsity=(
                "sparsity",
                "mean",
            ),

            mean_compression_ratio=(
                "compression_ratio",
                "mean",
            ),
        )
    )

    return aggregated


# ==========================================================
# CRITERION NORMALIZATION
# ==========================================================

def min_max_benefit(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize a benefit criterion.

    Higher original values are better.
    """

    minimum = series.min()
    maximum = series.max()

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
        )

    return (
        (series - minimum)
        / (maximum - minimum)
    )


def min_max_cost(
    series: pd.Series,
) -> pd.Series:
    """
    Normalize a cost criterion.

    Lower original values are better.
    """

    minimum = series.min()
    maximum = series.max()

    if np.isclose(
        maximum,
        minimum,
    ):
        return pd.Series(
            np.ones(len(series)),
            index=series.index,
        )

    return (
        (maximum - series)
        / (maximum - minimum)
    )


# ==========================================================
# PARETO FRONT
# ==========================================================

def identify_pareto_front(
    score_matrix: np.ndarray,
) -> np.ndarray:
    """
    Identify Pareto-optimal candidate wavelets.

    All criteria must already be represented such that
    higher values indicate better performance.
    """

    values = np.asarray(
        score_matrix,
        dtype=np.float64,
    )

    if values.ndim != 2:
        raise ValueError(
            "Score matrix must be two-dimensional."
        )

    number_of_candidates = (
        values.shape[0]
    )

    pareto = np.ones(
        number_of_candidates,
        dtype=bool,
    )

    for candidate_i in range(
        number_of_candidates
    ):

        for candidate_j in range(
            number_of_candidates
        ):

            if candidate_i == candidate_j:
                continue

            dominates = (
                np.all(
                    values[candidate_j]
                    >= values[candidate_i]
                )
                and
                np.any(
                    values[candidate_j]
                    > values[candidate_i]
                )
            )

            if dominates:
                pareto[
                    candidate_i
                ] = False
                break

    return pareto


# ==========================================================
# CANDIDATE RANKING
# ==========================================================

def rank_candidate_wavelets(
    aggregated_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank candidate wavelets independently for each modality.

    Cost criteria
    -------------
    - Normalized reconstruction error
    - Energy loss
    - Normalized entropy

    Benefit criteria
    ----------------
    - Sparsity
    - Compression ratio

    Final ranking uses:
        1. Pareto membership
        2. Equal-weight multi-criteria score
        3. Normalized RMSE as tie-breaker
    """

    ranked_groups = []

    for modality, group in (
        aggregated_metrics.groupby(
            "modality"
        )
    ):

        group = (
            group
            .copy()
            .reset_index(drop=True)
        )

        # Lower is better.
        group["score_rmse"] = (
            min_max_cost(
                group[
                    "mean_normalized_rmse"
                ]
            )
        )

        group["score_energy"] = (
            min_max_cost(
                group[
                    "mean_energy_loss"
                ]
            )
        )

        group["score_entropy"] = (
            min_max_cost(
                group[
                    "mean_normalized_entropy"
                ]
            )
        )

        # Higher is better.
        group["score_sparsity"] = (
            min_max_benefit(
                group[
                    "mean_sparsity"
                ]
            )
        )

        group["score_compression"] = (
            min_max_benefit(
                group[
                    "mean_compression_ratio"
                ]
            )
        )

        score_columns = [
            "score_rmse",
            "score_energy",
            "score_entropy",
            "score_sparsity",
            "score_compression",
        ]

        group["final_score"] = (
            group[
                score_columns
            ].mean(
                axis=1
            )
        )

        group["pareto_optimal"] = (
            identify_pareto_front(
                group[
                    score_columns
                ].to_numpy()
            )
        )

        group = group.sort_values(
            by=[
                "pareto_optimal",
                "final_score",
                "mean_normalized_rmse",
                "wavelet",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        ).reset_index(
            drop=True
        )

        group["rank"] = (
            np.arange(
                1,
                len(group) + 1,
            )
        )

        group["selected"] = False

        group.loc[
            0,
            "selected",
        ] = True

        ranked_groups.append(
            group
        )

    if not ranked_groups:
        return pd.DataFrame()

    return pd.concat(
        ranked_groups,
        ignore_index=True,
    )


# ==========================================================
# EXTRACT SELECTED WAVELETS
# ==========================================================

def extract_selected_wavelets(
    ranked_metrics: pd.DataFrame,
) -> Dict[str, Dict[str, Union[str, int, float, bool]]]:
    """
    Convert ranked results into a modality-specific dictionary.
    """

    if "selected" not in ranked_metrics.columns:
        raise ValueError(
            "Ranked metrics must contain a 'selected' column."
        )

    selected = (
        ranked_metrics[
            ranked_metrics[
                "selected"
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    results = {}

    for _, row in selected.iterrows():

        modality = str(
            row["modality"]
        )

        results[modality] = {
            "wavelet":
                str(row["wavelet"]),

            "level":
                int(
                    round(
                        row[
                            "decomposition_level"
                        ]
                    )
                ),

            "final_score":
                float(
                    row["final_score"]
                ),

            "pareto_optimal":
                bool(
                    row["pareto_optimal"]
                ),

            "mean_normalized_rmse":
                float(
                    row[
                        "mean_normalized_rmse"
                    ]
                ),

            "mean_energy_preservation":
                float(
                    row[
                        "mean_energy_preservation"
                    ]
                ),

            "mean_normalized_entropy":
                float(
                    row[
                        "mean_normalized_entropy"
                    ]
                ),

            "mean_sparsity":
                float(
                    row[
                        "mean_sparsity"
                    ]
                ),

            "mean_compression_ratio":
                float(
                    row[
                        "mean_compression_ratio"
                    ]
                ),
        }

    return results
