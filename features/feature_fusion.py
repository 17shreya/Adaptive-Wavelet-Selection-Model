"""
Multimodal feature fusion.


Selected features from ECG, trapezius EMG, and GSR are concatenated
horizontally to create a single multimodal feature representation.

Canonical modality order
------------------------
1. ECG
2. Trapezius EMG
3. GSR

Fusion method
-------------
Early feature concatenation.

The module:

- verifies modality feature matrices,
- checks sample alignment,
- ensures unique modality-specific feature names,
- concatenates selected modality features,
- validates NaN/Inf values,
- enforces the same feature schema across training and held-out data,
- optionally attaches sample metadata.

No dataset paths, CSV writing, or experiment-specific output
directories are included.

Author
------
Shreya
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ==========================================================
# DEFAULT CONFIGURATION
# ==========================================================

FUSION_METHOD = "early_feature_concatenation"

DEFAULT_MODALITY_ORDER = (
    "ecg",
    "emg_trapezius",
    "gsr",
)

DEFAULT_METADATA_COLUMNS = (
    "sample_key",
    "subject_id",
    "pain_class",
    "label",
)


# ==========================================================
# MODALITY ALIASES
# ==========================================================

MODALITY_ALIASES = {
    "ecg": "ecg",

    "emg": "emg_trapezius",
    "emg_trapezius": "emg_trapezius",

    "gsr": "gsr",
    "eda": "gsr",
}


PREFIX_ALIASES = {
    "ecg": (
        "ecg",
    ),

    "emg_trapezius": (
        "emg_trapezius",
        "emg",
    ),

    "gsr": (
        "gsr",
        "eda",
    ),
}


# ==========================================================
# MODALITY NAME NORMALIZATION
# ==========================================================

def canonicalize_modality_name(
    modality: str,
) -> str:
    """
    Convert a modality alias to the canonical repository name.

    Examples
    --------
    ``emg`` -> ``emg_trapezius``

    ``eda`` -> ``gsr``
    """

    modality = str(
        modality
    ).strip().lower()

    if modality not in MODALITY_ALIASES:

        raise ValueError(
            f"Unknown modality '{modality}'. "
            f"Supported modalities are: "
            f"{sorted(MODALITY_ALIASES)}"
        )

    return MODALITY_ALIASES[
        modality
    ]


# ==========================================================
# FEATURE MATRIX VALIDATION
# ==========================================================

def validate_feature_matrix(
    dataframe: pd.DataFrame,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    """
    Validate one modality-specific feature matrix.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Selected numerical feature matrix.

    modality : str, optional
        Modality name used in error messages.

    Returns
    -------
    pandas.DataFrame
        Validated feature matrix.

    Raises
    ------
    TypeError
        If input is not a DataFrame.

    ValueError
        If data are empty, non-numeric, contain duplicated
        feature names, NaN, or infinite values.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            "Feature matrix must be a pandas DataFrame."
        )

    if dataframe.empty:

        raise ValueError(
            f"{modality or 'Feature'} matrix is empty."
        )

    duplicate_columns = (
        dataframe.columns[
            dataframe.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            f"{modality or 'Feature'} matrix contains "
            f"duplicate columns: "
            f"{duplicate_columns[:10]}"
        )

    non_numeric_columns = [
        column
        for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if non_numeric_columns:

        raise ValueError(
            f"{modality or 'Feature'} matrix contains "
            f"non-numeric columns: "
            f"{non_numeric_columns[:10]}"
        )

    values = dataframe.to_numpy(
        dtype=np.float64
    )

    if np.isnan(
        values
    ).any():

        raise ValueError(
            f"NaN values detected in "
            f"{modality or 'feature'} matrix."
        )

    if np.isinf(
        values
    ).any():

        raise ValueError(
            f"Infinite values detected in "
            f"{modality or 'feature'} matrix."
        )

    return dataframe


# ==========================================================
# NORMALIZE MODALITY DICTIONARY
# ==========================================================

def canonicalize_modality_mapping(
    modality_matrices: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Convert modality dictionary keys to canonical names.

    Example
    -------
    ``{"ecg": ..., "emg": ..., "gsr": ...}``

    becomes

    ``{"ecg": ..., "emg_trapezius": ..., "gsr": ...}``
    """

    canonical = {}

    for modality, dataframe in (
        modality_matrices.items()
    ):

        canonical_name = (
            canonicalize_modality_name(
                modality
            )
        )

        if canonical_name in canonical:

            raise ValueError(
                f"Multiple matrices were provided for "
                f"modality '{canonical_name}'."
            )

        canonical[
            canonical_name
        ] = dataframe

    return canonical


# ==========================================================
# MODALITY PREFIXING
# ==========================================================

def ensure_modality_prefix(
    dataframe: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    """
    Ensure that all feature columns have a canonical modality prefix.

    Examples
    --------
    ECG feature:

        A6_energy
            ->
        ecg_A6_energy

    Older EMG feature:

        emg_A5_energy
            ->
        emg_trapezius_A5_energy

    Existing canonical feature:

        emg_trapezius_D5_rms
            ->
        emg_trapezius_D5_rms
    """

    modality = (
        canonicalize_modality_name(
            modality
        )
    )

    aliases = PREFIX_ALIASES[
        modality
    ]

    renamed_columns = {}

    for column in dataframe.columns:

        original_column = str(
            column
        )

        lower_column = (
            original_column.lower()
        )

        canonical_prefix = (
            f"{modality}_"
        )

        # Already canonical.
        if lower_column.startswith(
            canonical_prefix
        ):

            renamed_columns[
                column
            ] = original_column

            continue

        renamed = None

        # Convert old modality prefix to canonical prefix.
        for alias in aliases:

            alias_prefix = (
                f"{alias}_"
            )

            if lower_column.startswith(
                alias_prefix
            ):

                suffix = (
                    original_column[
                        len(alias_prefix):
                    ]
                )

                renamed = (
                    f"{modality}_"
                    f"{suffix}"
                )

                break

        # No prefix found.
        if renamed is None:

            renamed = (
                f"{modality}_"
                f"{original_column}"
            )

        renamed_columns[
            column
        ] = renamed

    output = dataframe.rename(
        columns=renamed_columns
    )

    duplicate_columns = (
        output.columns[
            output.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            "Duplicate columns were created during "
            f"{modality} prefixing: "
            f"{duplicate_columns[:10]}"
        )

    return output


# ==========================================================
# SAMPLE ALIGNMENT VALIDATION
# ==========================================================

def validate_modality_alignment(
    modality_matrices: Mapping[str, pd.DataFrame],
    require_same_index: bool = True,
) -> None:
    """
    Verify sample alignment across physiological modalities.

    All matrices must contain the same number of samples.

    If ``require_same_index`` is True, the DataFrame indices must
    also be identical.
    """

    if not modality_matrices:

        raise ValueError(
            "No modality matrices were provided."
        )

    modalities = list(
        modality_matrices.keys()
    )

    reference_modality = (
        modalities[0]
    )

    reference_dataframe = (
        modality_matrices[
            reference_modality
        ]
    )

    reference_length = len(
        reference_dataframe
    )

    reference_index = (
        reference_dataframe.index
    )

    for modality in modalities[1:]:

        dataframe = (
            modality_matrices[
                modality
            ]
        )

        if len(
            dataframe
        ) != reference_length:

            raise ValueError(
                "Sample-count mismatch between "
                f"{reference_modality} "
                f"({reference_length}) and "
                f"{modality} "
                f"({len(dataframe)})."
            )

        if (
            require_same_index
            and not dataframe.index.equals(
                reference_index
            )
        ):

            raise ValueError(
                "Sample indices are not aligned between "
                f"{reference_modality} and {modality}."
            )


# ==========================================================
# FEATURE MATRIX FUSION
# ==========================================================

def fuse_feature_matrices(
    modality_matrices: Mapping[str, pd.DataFrame],
    modality_order: Sequence[str] = DEFAULT_MODALITY_ORDER,
    add_modality_prefix: bool = True,
    require_same_index: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Concatenate selected physiological feature matrices.

    Parameters
    ----------
    modality_matrices : mapping
        Mapping from modality name to selected feature DataFrame.

    modality_order : sequence of str, optional
        Order used during concatenation.

    add_modality_prefix : bool, optional
        Add canonical modality prefixes to feature names.

    require_same_index : bool, optional
        Require identical sample indices across modalities.

    Returns
    -------
    fused_features : pandas.DataFrame
        Concatenated multimodal feature matrix.

    modality_feature_counts : dict
        Number of features contributed by each modality.
    """

    matrices = (
        canonicalize_modality_mapping(
            modality_matrices
        )
    )

    canonical_order = [
        canonicalize_modality_name(
            modality
        )
        for modality in modality_order
    ]

    missing_modalities = [
        modality
        for modality in canonical_order
        if modality not in matrices
    ]

    if missing_modalities:

        raise ValueError(
            "Missing modality matrices: "
            f"{missing_modalities}"
        )

    validated_matrices = {}

    for modality in canonical_order:

        dataframe = (
            matrices[
                modality
            ].copy()
        )

        validate_feature_matrix(
            dataframe,
            modality=modality,
        )

        if add_modality_prefix:

            dataframe = (
                ensure_modality_prefix(
                    dataframe,
                    modality,
                )
            )

        validated_matrices[
            modality
        ] = dataframe

    validate_modality_alignment(
        validated_matrices,
        require_same_index=
            require_same_index,
    )

    modality_feature_counts = {
        modality:
            validated_matrices[
                modality
            ].shape[1]
        for modality in canonical_order
    }

    fused_features = pd.concat(
        [
            validated_matrices[
                modality
            ]
            for modality
            in canonical_order
        ],
        axis=1,
    )

    duplicate_columns = (
        fused_features.columns[
            fused_features.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            "Duplicate feature names detected "
            "after multimodal fusion: "
            f"{duplicate_columns[:10]}"
        )

    validate_feature_matrix(
        fused_features,
        modality="fused",
    )

    expected_feature_count = sum(
        modality_feature_counts.values()
    )

    if (
        fused_features.shape[1]
        != expected_feature_count
    ):

        raise RuntimeError(
            "Unexpected fused feature dimension."
        )

    return (
        fused_features,
        modality_feature_counts,
    )


# ==========================================================
# ATTACH METADATA
# ==========================================================

def attach_metadata(
    fused_features: pd.DataFrame,
    metadata: pd.DataFrame,
    metadata_columns: Optional[Sequence[str]] = None,
    require_same_index: bool = True,
) -> pd.DataFrame:
    """
    Attach sample metadata to a fused feature matrix.

    Parameters
    ----------
    fused_features : pandas.DataFrame
        Fused numerical features.

    metadata : pandas.DataFrame
        Metadata table.

    metadata_columns : sequence of str, optional
        Metadata columns to retain.

    require_same_index : bool, optional
        Require metadata and features to have identical indices.

    Returns
    -------
    pandas.DataFrame
        Metadata followed by fused features.
    """

    if not isinstance(
        metadata,
        pd.DataFrame,
    ):

        raise TypeError(
            "Metadata must be a pandas DataFrame."
        )

    if len(
        metadata
    ) != len(
        fused_features
    ):

        raise ValueError(
            "Metadata and fused features have "
            "different sample counts."
        )

    if (
        require_same_index
        and not metadata.index.equals(
            fused_features.index
        )
    ):

        raise ValueError(
            "Metadata and fused features "
            "are not sample aligned."
        )

    if metadata_columns is not None:

        missing_columns = [
            column
            for column in metadata_columns
            if column not in metadata.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing metadata columns: "
                f"{missing_columns}"
            )

        metadata = metadata[
            list(
                metadata_columns
            )
        ].copy()

    else:

        metadata = metadata.copy()

    if metadata.isna().any().any():

        raise ValueError(
            "Missing values detected in metadata."
        )

    return pd.concat(
        [
            metadata,
            fused_features,
        ],
        axis=1,
    )


# ==========================================================
# COMPLETE MULTIMODAL FUSION
# ==========================================================

def fuse_modalities(
    modality_matrices: Mapping[str, pd.DataFrame],
    metadata: Optional[pd.DataFrame] = None,
    modality_order: Sequence[str] = DEFAULT_MODALITY_ORDER,
    metadata_columns: Optional[Sequence[str]] = None,
    require_same_index: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Perform complete early multimodal feature fusion.

    Parameters
    ----------
    modality_matrices : mapping
        Selected ECG, trapezius EMG, and GSR feature matrices.

    metadata : pandas.DataFrame, optional
        Sample metadata.

    modality_order : sequence of str, optional
        Modality concatenation order.

    metadata_columns : sequence of str, optional
        Metadata columns retained in the final DataFrame.

    require_same_index : bool, optional
        Require sample indices to match across modalities.

    Returns
    -------
    fused_dataframe : pandas.DataFrame
        Fused features, optionally including metadata.

    modality_feature_counts : dict
        Feature count contributed by each modality.
    """

    (
        fused_features,
        feature_counts,
    ) = fuse_feature_matrices(
        modality_matrices=
            modality_matrices,

        modality_order=
            modality_order,

        add_modality_prefix=True,

        require_same_index=
            require_same_index,
    )

    if metadata is None:

        return (
            fused_features,
            feature_counts,
        )

    fused_dataframe = (
        attach_metadata(
            fused_features=
                fused_features,

            metadata=
                metadata,

            metadata_columns=
                metadata_columns,

            require_same_index=
                require_same_index,
        )
    )

    return (
        fused_dataframe,
        feature_counts,
    )


# ==========================================================
# FUSION FEATURE MANIFEST
# ==========================================================

def create_fusion_manifest(
    modality_matrices: Mapping[str, pd.DataFrame],
    ranked_features: Optional[
        Mapping[str, Sequence[str]]
    ] = None,
) -> pd.DataFrame:
    """
    Create a manifest describing all fused features.

    Parameters
    ----------
    modality_matrices : mapping
        Selected feature matrices.

    ranked_features : mapping, optional
        Selected features ordered by feature-selection rank.

    Returns
    -------
    pandas.DataFrame
        Fusion feature manifest.
    """

    matrices = (
        canonicalize_modality_mapping(
            modality_matrices
        )
    )

    records = []

    for modality in (
        DEFAULT_MODALITY_ORDER
    ):

        if modality not in matrices:
            continue

        original_features = (
            matrices[
                modality
            ].columns.tolist()
        )

        prefixed_dataframe = (
            ensure_modality_prefix(
                matrices[
                    modality
                ],
                modality,
            )
        )

        fused_features = (
            prefixed_dataframe
            .columns
            .tolist()
        )

        rank_lookup = {}

        if ranked_features is not None:

            candidate_keys = [
                modality
            ]

            if modality == "emg_trapezius":
                candidate_keys.append(
                    "emg"
                )

            ranking = None

            for key in candidate_keys:

                if key in ranked_features:

                    ranking = list(
                        ranked_features[
                            key
                        ]
                    )

                    break

            if ranking is not None:

                rank_lookup = {
                    feature: rank
                    for rank, feature
                    in enumerate(
                        ranking,
                        start=1,
                    )
                }

        for matrix_position, (
            original_feature,
            fused_feature,
        ) in enumerate(
            zip(
                original_features,
                fused_features,
            ),
            start=1,
        ):

            records.append(
                {
                    "modality":
                        modality,

                    "matrix_position":
                        matrix_position,

                    "selection_rank":
                        rank_lookup.get(
                            original_feature
                        ),

                    "original_feature_name":
                        original_feature,

                    "fused_feature_name":
                        fused_feature,
                }
            )

    return pd.DataFrame(
        records
    )


# ==========================================================
# REUSABLE TRAIN/FOLD FUSION TRANSFORMER
# ==========================================================

class EarlyFeatureFusion:
    """
    Reusable early-feature fusion transformer.

    ``fit`` records the modality-specific feature schema from
    training data.

    ``transform`` requires validation/test data to contain exactly
    the same selected features, preventing accidental feature-set
    inconsistencies between subject-independent folds.
    """

    def __init__(
        self,
        modality_order: Sequence[str] = DEFAULT_MODALITY_ORDER,
        require_same_index: bool = True,
    ):

        self.modality_order = tuple(
            canonicalize_modality_name(
                modality
            )
            for modality in modality_order
        )

        self.require_same_index = (
            require_same_index
        )

        self.input_features_ = None

        self.output_features_ = None

        self.feature_counts_ = None


    def fit(
        self,
        modality_matrices: Mapping[
            str,
            pd.DataFrame,
        ],
    ) -> "EarlyFeatureFusion":
        """
        Record the selected feature schema from training data.
        """

        matrices = (
            canonicalize_modality_mapping(
                modality_matrices
            )
        )

        (
            fused,
            feature_counts,
        ) = fuse_feature_matrices(
            modality_matrices=
                matrices,

            modality_order=
                self.modality_order,

            require_same_index=
                self.require_same_index,
        )

        self.input_features_ = {
            modality:
                matrices[
                    modality
                ].columns.tolist()
            for modality
            in self.modality_order
        }

        self.output_features_ = (
            fused.columns.tolist()
        )

        self.feature_counts_ = (
            feature_counts
        )

        return self


    def transform(
        self,
        modality_matrices: Mapping[
            str,
            pd.DataFrame,
        ],
    ) -> pd.DataFrame:
        """
        Fuse held-out data using the feature schema learned on training.
        """

        if self.input_features_ is None:

            raise RuntimeError(
                "EarlyFeatureFusion must be fitted "
                "before transform()."
            )

        matrices = (
            canonicalize_modality_mapping(
                modality_matrices
            )
        )

        prepared = {}

        for modality in (
            self.modality_order
        ):

            if modality not in matrices:

                raise ValueError(
                    f"Missing modality: "
                    f"{modality}"
                )

            dataframe = (
                matrices[
                    modality
                ].copy()
            )

            expected_features = (
                self.input_features_[
                    modality
                ]
            )

            missing_features = [
                feature
                for feature
                in expected_features
                if feature not in
                dataframe.columns
            ]

            if missing_features:

                raise ValueError(
                    f"{modality} is missing "
                    "training-selected features: "
                    f"{missing_features[:10]}"
                )

            prepared[
                modality
            ] = dataframe[
                expected_features
            ].copy()

        fused, _ = (
            fuse_feature_matrices(
                modality_matrices=
                    prepared,

                modality_order=
                    self.modality_order,

                require_same_index=
                    self.require_same_index,
            )
        )

        if fused.columns.tolist() != (
            self.output_features_
        ):

            raise RuntimeError(
                "Fused feature schema differs "
                "from the training schema."
            )

        return fused


    def fit_transform(
        self,
        modality_matrices: Mapping[
            str,
            pd.DataFrame,
        ],
    ) -> pd.DataFrame:
        """
        Fit fusion schema and fuse training data.
        """

        self.fit(
            modality_matrices
        )

        return self.transform(
            modality_matrices
        )


    def get_feature_names_out(
        self,
    ) -> List[str]:
        """
        Return final fused feature names.
        """

        if self.output_features_ is None:

            raise RuntimeError(
                "Fusion transformer has not been fitted."
            )

        return list(
            self.output_features_
        )


    def get_feature_counts(
        self,
    ) -> Dict[str, int]:
        """
        Return modality-specific selected feature counts.
        """

        if self.feature_counts_ is None:

            raise RuntimeError(
                "Fusion transformer has not been fitted."
            )

        return dict(
            self.feature_counts_
        )



"""
Cross-modal physiological feature fusion.

This module implements direct feature-level fusion for multimodal
physiological pain recognition.

Supported representations
-------------------------
Unimodal:
    ECG
    Trapezius EMG
    GSR

Bimodal:
    ECG + EMG
    ECG + GSR
    EMG + GSR

Trimodal:
    ECG + EMG + GSR



"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ==========================================================
# FUSION CONFIGURATION
# ==========================================================

FUSION_METHOD = "direct_feature_concatenation"


# Canonical modality names used inside the repository.
DEFAULT_MODALITY_ORDER = (
    "ecg",
    "emg_trapezius",
    "gsr",
)


# ==========================================================
# CROSS-MODAL REPRESENTATIONS
# ==========================================================

REPRESENTATION_DEFINITIONS = {

    # Unimodal
    "ecg": (
        "ecg",
    ),

    "emg": (
        "emg_trapezius",
    ),

    "gsr": (
        "gsr",
    ),

    # Bimodal
    "ecg_emg": (
        "ecg",
        "emg_trapezius",
    ),

    "ecg_gsr": (
        "ecg",
        "gsr",
    ),

    "emg_gsr": (
        "emg_trapezius",
        "gsr",
    ),

    # Trimodal
    "fusion": (
        "ecg",
        "emg_trapezius",
        "gsr",
    ),
}


DISPLAY_NAMES = {

    "ecg":
        "ECG",

    "emg":
        "EMG",

    "gsr":
        "GSR",

    "ecg_emg":
        "ECG + EMG",

    "ecg_gsr":
        "ECG + GSR",

    "emg_gsr":
        "EMG + GSR",

    "fusion":
        "ECG + EMG + GSR",
}


# ==========================================================
# MODALITY ALIASES
# ==========================================================

MODALITY_ALIASES = {

    "ecg":
        "ecg",

    "emg":
        "emg_trapezius",

    "emg_trapezius":
        "emg_trapezius",

    "gsr":
        "gsr",

    "eda":
        "gsr",
}


# ==========================================================
# CANONICAL MODALITY NAME
# ==========================================================

def canonicalize_modality_name(
    modality: str,
) -> str:
    """
    Convert a modality alias to the canonical repository name.

    Examples
    --------
    emg -> emg_trapezius

    eda -> gsr
    """

    modality = str(
        modality
    ).strip().lower()

    if modality not in MODALITY_ALIASES:

        raise ValueError(
            f"Unknown modality: {modality}"
        )

    return MODALITY_ALIASES[
        modality
    ]


# ==========================================================
# CANONICALIZE MODALITY DICTIONARY
# ==========================================================

def canonicalize_modality_mapping(
    modality_matrices: Mapping[
        str,
        pd.DataFrame,
    ],
) -> Dict[str, pd.DataFrame]:
    """
    Convert dictionary keys to canonical modality names.
    """

    canonical = {}

    for modality, dataframe in (
        modality_matrices.items()
    ):

        canonical_name = (
            canonicalize_modality_name(
                modality
            )
        )

        if canonical_name in canonical:

            raise ValueError(
                "Duplicate modality provided: "
                f"{canonical_name}"
            )

        canonical[
            canonical_name
        ] = dataframe

    return canonical


# ==========================================================
# FEATURE MATRIX VALIDATION
# ==========================================================

def validate_feature_matrix(
    dataframe: pd.DataFrame,
    modality: Optional[str] = None,
) -> None:
    """
    Validate one modality-specific feature matrix.

    Checks
    ------
    - DataFrame is not empty
    - Features are numeric
    - No duplicated columns
    - No NaN values
    - No infinite values
    """

    name = (
        modality
        if modality is not None
        else "feature"
    )

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            f"{name} matrix must be a pandas DataFrame."
        )

    if dataframe.empty:

        raise ValueError(
            f"{name} feature matrix is empty."
        )

    duplicate_columns = (
        dataframe.columns[
            dataframe.columns.duplicated()
        ]
        .tolist()
    )

    if duplicate_columns:

        raise ValueError(
            f"{name} contains duplicate feature names: "
            f"{duplicate_columns[:10]}"
        )

    non_numeric = [
        column
        for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]

    if non_numeric:

        raise ValueError(
            f"{name} contains non-numeric feature columns: "
            f"{non_numeric[:10]}"
        )

    values = dataframe.to_numpy(
        dtype=np.float64
    )

    if np.isnan(
        values
    ).any():

        raise ValueError(
            f"{name} contains NaN values."
        )

    if np.isinf(
        values
    ).any():

        raise ValueError(
            f"{name} contains infinite values."
        )


# ==========================================================
# MODALITY-SPECIFIC FEATURE PREFIX
# ==========================================================

def ensure_modality_prefix(
    dataframe: pd.DataFrame,
    modality: str,
) -> pd.DataFrame:
    """
    Ensure that every feature contains a modality prefix.

    Examples
    --------
    A6_energy
        ->
    ecg_A6_energy

    emg_A5_rms
        ->
    emg_trapezius_A5_rms
    """

    modality = (
        canonicalize_modality_name(
            modality
        )
    )

    output = dataframe.copy()

    renamed_columns = {}

    for column in output.columns:

        column_string = str(
            column
        )

        column_lower = (
            column_string.lower()
        )

        canonical_prefix = (
            f"{modality}_"
        )

        # Already correctly prefixed.
        if column_lower.startswith(
            canonical_prefix
        ):

            renamed_columns[
                column
            ] = column_string

            continue

        # Convert old EMG prefix.
        if (
            modality == "emg_trapezius"
            and column_lower.startswith(
                "emg_"
            )
        ):

            suffix = column_string[
                len("emg_"):
            ]

            renamed_columns[
                column
            ] = (
                f"emg_trapezius_"
                f"{suffix}"
            )

            continue

        # Convert EDA prefix to GSR.
        if (
            modality == "gsr"
            and column_lower.startswith(
                "eda_"
            )
        ):

            suffix = column_string[
                len("eda_"):
            ]

            renamed_columns[
                column
            ] = (
                f"gsr_"
                f"{suffix}"
            )

            continue

        renamed_columns[
            column
        ] = (
            f"{modality}_"
            f"{column_string}"
        )

    output = output.rename(
        columns=renamed_columns
    )

    if output.columns.duplicated().any():

        duplicates = (
            output.columns[
                output.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate feature names generated "
            f"after prefixing: {duplicates[:10]}"
        )

    return output


# ==========================================================
# SAMPLE ALIGNMENT
# ==========================================================

def validate_modality_alignment(
    modality_matrices: Mapping[
        str,
        pd.DataFrame,
    ],
    require_same_index: bool = True,
) -> None:
    """
    Verify that modality matrices represent the same samples.

    All matrices must contain the same number of rows.

    If ``require_same_index`` is True, DataFrame indices must
    also match exactly.
    """

    if not modality_matrices:

        raise ValueError(
            "No modality matrices were provided."
        )

    modalities = list(
        modality_matrices.keys()
    )

    reference_modality = (
        modalities[0]
    )

    reference_dataframe = (
        modality_matrices[
            reference_modality
        ]
    )

    reference_length = len(
        reference_dataframe
    )

    reference_index = (
        reference_dataframe.index
    )

    for modality in modalities[1:]:

        dataframe = (
            modality_matrices[
                modality
            ]
        )

        if len(
            dataframe
        ) != reference_length:

            raise ValueError(
                "Sample-count mismatch: "
                f"{reference_modality}="
                f"{reference_length}, "
                f"{modality}="
                f"{len(dataframe)}"
            )

        if (
            require_same_index
            and not dataframe.index.equals(
                reference_index
            )
        ):

            raise ValueError(
                "Sample ordering mismatch between "
                f"{reference_modality} and {modality}."
            )


# ==========================================================
# METADATA ALIGNMENT
# ==========================================================

def validate_metadata_alignment(
    modality_metadata: Mapping[
        str,
        pd.DataFrame,
    ],
    alignment_columns: Sequence[str] = (
        "subject_id",
        "pain_class",
        "trial_id",
        "segment_id",
    ),
) -> None:
    """
    Verify physiological modalities represent identical samples.

    Alignment is checked using available metadata columns.

    Parameters
    ----------
    modality_metadata : mapping
        Metadata DataFrame for each modality.

    alignment_columns : sequence of str
        Metadata identifiers used for sample alignment.
    """

    metadata = (
        canonicalize_modality_mapping(
            modality_metadata
        )
    )

    modalities = list(
        metadata.keys()
    )

    if len(
        modalities
    ) < 2:

        return

    reference = metadata[
        modalities[0]
    ].reset_index(
        drop=True
    )

    for modality in modalities[1:]:

        current = metadata[
            modality
        ].reset_index(
            drop=True
        )

        if len(
            current
        ) != len(
            reference
        ):

            raise ValueError(
                f"Metadata sample-count mismatch "
                f"for {modality}."
            )

        for column in alignment_columns:

            if (
                column in reference.columns
                and column in current.columns
            ):

                reference_values = (
                    reference[
                        column
                    ]
                    .astype(str)
                    .to_numpy()
                )

                current_values = (
                    current[
                        column
                    ]
                    .astype(str)
                    .to_numpy()
                )

                if not np.array_equal(
                    reference_values,
                    current_values,
                ):

                    raise ValueError(
                        f"Cross-modal mismatch in "
                        f"'{column}' between "
                        f"{modalities[0]} and "
                        f"{modality}."
                    )


# ==========================================================
# DIRECT CROSS-MODAL CONCATENATION
# ==========================================================

def concatenate_modalities(
    modality_matrices: Mapping[
        str,
        pd.DataFrame,
    ],
    modalities: Sequence[str],
    require_same_index: bool = True,
) -> Tuple[
    pd.DataFrame,
    Dict[str, int],
]:
    """
    Directly concatenate selected features from multiple modalities.

    No attention, weighting, or learned fusion is performed.

    Parameters
    ----------
    modality_matrices : mapping
        Selected feature matrices.

    modalities : sequence of str
        Modalities included in the representation.

    require_same_index : bool
        Require identical row indices.

    Returns
    -------
    fused_dataframe : pandas.DataFrame
        Directly concatenated feature representation.

    feature_counts : dict
        Number of features contributed by each modality.
    """

    matrices = (
        canonicalize_modality_mapping(
            modality_matrices
        )
    )

    requested_modalities = [
        canonicalize_modality_name(
            modality
        )
        for modality in modalities
    ]

    missing = [
        modality
        for modality
        in requested_modalities
        if modality not in matrices
    ]

    if missing:

        raise ValueError(
            f"Missing modality matrices: {missing}"
        )

    prepared = {}

    feature_counts = {}

    for modality in (
        requested_modalities
    ):

        dataframe = (
            matrices[
                modality
            ]
            .copy()
        )

        validate_feature_matrix(
            dataframe,
            modality=modality,
        )

        dataframe = (
            ensure_modality_prefix(
                dataframe,
                modality,
            )
        )

        prepared[
            modality
        ] = dataframe

        feature_counts[
            modality
        ] = (
            dataframe.shape[1]
        )

    validate_modality_alignment(
        prepared,
        require_same_index=
            require_same_index,
    )

    fused_dataframe = pd.concat(
        [
            prepared[
                modality
            ]
            for modality
            in requested_modalities
        ],
        axis=1,
    )

    if fused_dataframe.columns.duplicated().any():

        duplicates = (
            fused_dataframe.columns[
                fused_dataframe.columns.duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            "Duplicate feature names detected after "
            f"fusion: {duplicates[:10]}"
        )

    validate_feature_matrix(
        fused_dataframe,
        modality="cross-modal fusion",
    )

    expected_features = sum(
        feature_counts.values()
    )

    if (
        fused_dataframe.shape[1]
        != expected_features
    ):

        raise RuntimeError(
            "Cross-modal feature-count mismatch."
        )

    return (
        fused_dataframe,
        feature_counts,
    )


# ==========================================================
# CREATE ALL CROSS-MODAL REPRESENTATIONS
# ==========================================================

def build_cross_modal_representations(
    modality_matrices: Mapping[
        str,
        pd.DataFrame,
    ],
    require_same_index: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Create all unimodal, bimodal, and trimodal representations.

    Representations
    ---------------
    ecg
    emg
    gsr
    ecg_emg
    ecg_gsr
    emg_gsr
    fusion
    """

    matrices = (
        canonicalize_modality_mapping(
            modality_matrices
        )
    )

    representations = {}

    for (
        representation_name,
        modalities,
    ) in REPRESENTATION_DEFINITIONS.items():

        fused, _ = (
            concatenate_modalities(
                modality_matrices=
                    matrices,

                modalities=
                    modalities,

                require_same_index=
                    require_same_index,
            )
        )

        representations[
            representation_name
        ] = fused

    return representations


# ==========================================================
# FEATURE-COUNT AUDIT
# ==========================================================

def create_dimension_report(
    representations: Mapping[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Create a compact dimensionality report.
    """

    records = []

    for (
        representation_name,
        dataframe,
    ) in representations.items():

        records.append(
            {
                "representation":
                    representation_name,

                "representation_name":
                    DISPLAY_NAMES.get(
                        representation_name,
                        representation_name,
                    ),

                "samples":
                    dataframe.shape[0],

                "features":
                    dataframe.shape[1],
            }
        )

    return pd.DataFrame(
        records
    )


# ==========================================================
# MATHEMATICAL FEATURE-COUNT CHECK
# ==========================================================

def validate_cross_modal_dimensions(
    representations: Mapping[
        str,
        pd.DataFrame,
    ],
) -> None:
    """
    Verify mathematical feature counts for all fused representations.
    """

    required = {
        "ecg",
        "emg",
        "gsr",
        "ecg_emg",
        "ecg_gsr",
        "emg_gsr",
        "fusion",
    }

    missing = (
        required
        - set(
            representations.keys()
        )
    )

    if missing:

        raise ValueError(
            "Missing representations: "
            f"{sorted(missing)}"
        )

    ecg_features = (
        representations[
            "ecg"
        ].shape[1]
    )

    emg_features = (
        representations[
            "emg"
        ].shape[1]
    )

    gsr_features = (
        representations[
            "gsr"
        ].shape[1]
    )

    expected = {

        "ecg_emg":
            ecg_features
            + emg_features,

        "ecg_gsr":
            ecg_features
            + gsr_features,

        "emg_gsr":
            emg_features
            + gsr_features,

        "fusion":
            ecg_features
            + emg_features
            + gsr_features,
    }

    for (
        representation,
        expected_count,
    ) in expected.items():

        actual_count = (
            representations[
                representation
            ].shape[1]
        )

        if actual_count != expected_count:

            raise RuntimeError(
                f"{representation}: expected "
                f"{expected_count} features but "
                f"found {actual_count}."
            )


# ==========================================================
# CROSS-MODAL FUSION TRANSFORMER
# ==========================================================

class CrossModalFeatureFusion:
    """
    Cross-modal direct feature-concatenation transformer.

    The transformer records the training feature schema and ensures
    exactly the same selected features are used for validation/test
    subjects.

    This prevents accidental feature-schema inconsistencies during
    subject-independent evaluation.
    """

    def __init__(
        self,
        require_same_index: bool = True,
    ):

        self.require_same_index = (
            require_same_index
        )

        self.training_feature_schema_ = None

        self.output_feature_schema_ = None


    def fit(
        self,
        modality_matrices: Mapping[
            str,
            pd.DataFrame,
        ],
    ) -> "CrossModalFeatureFusion":
        """
        Learn feature schema from training data.
        """

        matrices = (
            canonicalize_modality_mapping(
                modality_matrices
            )
        )

        required_modalities = {
            "ecg",
            "emg_trapezius",
            "gsr",
        }

        missing = (
            required_modalities
            - set(
                matrices.keys()
            )
        )

        if missing:

            raise ValueError(
                "Missing training modalities: "
                f"{sorted(missing)}"
            )

        self.training_feature_schema_ = {

            modality:
                matrices[
                    modality
                ].columns.tolist()

            for modality
            in DEFAULT_MODALITY_ORDER
        }

        representations = (
            build_cross_modal_representations(
                modality_matrices=
                    matrices,

                require_same_index=
                    self.require_same_index,
            )
        )

        validate_cross_modal_dimensions(
            representations
        )

        self.output_feature_schema_ = {

            representation:
                dataframe.columns.tolist()

            for (
                representation,
                dataframe,
            ) in representations.items()
        }

        return self


    def transform(
        self,
        modality_matrices: Mapping[
            str,
            pd.DataFrame,
        ],
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all cross-modal representations using the
        training feature schema.
        """

        if self.training_feature_schema_ is None:

            raise RuntimeError(
                "CrossModalFeatureFusion must be fitted "
                "before transform()."
            )

        matrices = (
            canonicalize_modality_mapping(
                modality_matrices
            )
        )

        prepared = {}

        for modality in (
            DEFAULT_MODALITY_ORDER
        ):

            if modality not in matrices:

                raise ValueError(
                    f"Missing modality: {modality}"
                )

            expected_features = (
                self.training_feature_schema_[
                    modality
                ]
            )

            missing_features = [
                feature
                for feature
                in expected_features
                if feature
                not in matrices[
                    modality
                ].columns
            ]

            if missing_features:

                raise ValueError(
                    f"{modality} is missing "
                    "training-selected features: "
                    f"{missing_features[:10]}"
                )

            prepared[
                modality
            ] = matrices[
                modality
            ][
                expected_features
            ].copy()

        representations = (
            build_cross_modal_representations(
                modality_matrices=
                    prepared,

                require_same_index=
                    self.require_same_index,
            )
        )

        validate_cross_modal_dimensions(
            representations
        )

        # Verify feature schema against training.
        for (
            representation,
            dataframe,
        ) in representations.items():

            expected_columns = (
                self.output_feature_schema_[
                    representation
                ]
            )

            if (
                dataframe.columns.tolist()
                != expected_columns
            ):

                raise RuntimeError(
                    "Feature schema changed for "
                    f"{representation}."
                )

        return representations


    def fit_transform(
        self,
        modality_matrices: Mapping[
            str,
            pd.DataFrame,
        ],
    ) -> Dict[str, pd.DataFrame]:
        """
        Fit fusion schema and create training representations.
        """

        self.fit(
            modality_matrices
        )

        return self.transform(
            modality_matrices
        )

from pain_recognition.fusion.feature_fusion import (
    CrossModalFeatureFusion,
)


cross_modal_fusion = (
    CrossModalFeatureFusion()
)


train_representations = (
    cross_modal_fusion.fit_transform(
        {
            "ecg":
                X_train_ecg,

            "emg_trapezius":
                X_train_emg,

            "gsr":
                X_train_gsr,
        }
    )
)
