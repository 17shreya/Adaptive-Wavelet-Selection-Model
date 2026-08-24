"""
BioVid dataset utilities.

This module contains BioVid Heat Pain Database-specific functionality
used by the multimodal physiological pain-recognition framework.

Responsibilities
----------------
- BioVid class-label mapping
- Filename metadata parsing
- Subject ID extraction
- Pain-class extraction
- Trial ID extraction
- Segment ID extraction
- Sample-key generation
- Metadata validation
- Canonical label conversion

Canonical pain classes
----------------------
BL1 -> 0
PA1 -> 1
PA2 -> 2
PA3 -> 3
PA4 -> 4

This module does not perform preprocessing, feature extraction,
classification, or file-system traversal.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd


# ==========================================================
# BIOVID CLASS CONFIGURATION
# ==========================================================

BIOVID_CLASS_NAMES = (
    "BL1",
    "PA1",
    "PA2",
    "PA3",
    "PA4",
)


BIOVID_CLASS_LABELS = (
    0,
    1,
    2,
    3,
    4,
)


BIOVID_CLASS_MAPPING = {
    "BL1": 0,
    "PA1": 1,
    "PA2": 2,
    "PA3": 3,
    "PA4": 4,
}


BIOVID_REVERSE_CLASS_MAPPING = {
    value: key
    for key, value
    in BIOVID_CLASS_MAPPING.items()
}


# ==========================================================
# KNOWN METADATA COLUMNS
# ==========================================================

BIOVID_METADATA_COLUMNS = (
    "sample_key",
    "subject_id",
    "pain_class",
    "trial_id",
    "segment_id",
    "split",
    "label",
    "binary_label",
)


# ==========================================================
# LABEL NORMALIZATION
# ==========================================================

def canonicalize_biovid_label(
    value,
) -> int:
    """
    Convert one BioVid label to canonical integer format.

    Accepted examples
    -----------------
    BL1
    PA1
    PA2
    PA3
    PA4

    0
    1
    2
    3
    4

    "0"
    "1"

    "0.0"
    "1.0"

    Returns
    -------
    int
        Canonical label from 0 to 4.
    """

    # ------------------------------------------------------
    # Integer input
    # ------------------------------------------------------

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):

        label = int(
            value
        )

    # ------------------------------------------------------
    # Floating-point input
    # ------------------------------------------------------

    elif isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):

        if not np.isfinite(
            value
        ):

            raise ValueError(
                f"Non-finite BioVid label: {value}"
            )

        if not float(
            value
        ).is_integer():

            raise ValueError(
                f"BioVid label must be integer-valued: {value}"
            )

        label = int(
            value
        )

    # ------------------------------------------------------
    # String input
    # ------------------------------------------------------

    else:

        value_string = (
            str(
                value
            )
            .strip()
            .upper()
        )

        # Named class
        if value_string in (
            BIOVID_CLASS_MAPPING
        ):

            label = (
                BIOVID_CLASS_MAPPING[
                    value_string
                ]
            )

        # Numeric string
        else:

            try:

                numeric_value = float(
                    value_string
                )

            except ValueError as exc:

                raise ValueError(
                    f"Unknown BioVid label: {value!r}"
                ) from exc

            if not np.isfinite(
                numeric_value
            ):

                raise ValueError(
                    f"Non-finite BioVid label: {value}"
                )

            if not numeric_value.is_integer():

                raise ValueError(
                    f"Invalid BioVid numerical label: {value}"
                )

            label = int(
                numeric_value
            )

    # ------------------------------------------------------
    # Final validation
    # ------------------------------------------------------

    if label not in (
        BIOVID_CLASS_LABELS
    ):

        raise ValueError(
            f"BioVid label must be one of "
            f"{BIOVID_CLASS_LABELS}. "
            f"Received: {label}"
        )

    return label


# ==========================================================
# LABEL ARRAY NORMALIZATION
# ==========================================================

def canonicalize_biovid_labels(
    labels,
) -> np.ndarray:
    """
    Convert an array of BioVid labels to canonical integer labels.

    Returns
    -------
    numpy.ndarray
        Integer array with values from 0 to 4.
    """

    labels = np.asarray(
        labels
    ).reshape(-1)

    converted = [
        canonicalize_biovid_label(
            value
        )
        for value in labels
    ]

    return np.asarray(
        converted,
        dtype=np.int64,
    )


# ==========================================================
# LABEL TO CLASS NAME
# ==========================================================

def label_to_class_name(
    label,
) -> str:
    """
    Convert BioVid integer label to class name.

    Example
    -------
    0 -> BL1
    4 -> PA4
    """

    canonical_label = (
        canonicalize_biovid_label(
            label
        )
    )

    return BIOVID_REVERSE_CLASS_MAPPING[
        canonical_label
    ]


# ==========================================================
# CLASS NAME NORMALIZATION
# ==========================================================

def canonicalize_class_name(
    value,
) -> str:
    """
    Convert BioVid class representation to canonical class name.

    Example
    -------
    0 -> BL1
    "pa4" -> PA4
    """

    label = (
        canonicalize_biovid_label(
            value
        )
    )

    return label_to_class_name(
        label
    )


# ==========================================================
# FILENAME METADATA PARSER
# ==========================================================

def parse_biovid_filename(
    filename: Union[
        str,
        Path,
    ],
) -> Dict[str, Optional[Union[str, int]]]:
    """
    Parse BioVid metadata from a filename.

    Supported filename pattern
    --------------------------
    Example:

    071309_w_21-BL1-081_bio_segment_000_ecg_filtered_A6.csv

    Extracted information
    ---------------------
    subject_id
    pain_class
    trial_id
    segment_id
    modality
    coefficient
    coefficient_type
    coefficient_level

    Parameters
    ----------
    filename : str or pathlib.Path
        BioVid-derived filename.

    Returns
    -------
    dict
        Parsed metadata.
    """

    stem = Path(
        filename
    ).stem

    metadata = {
        "subject_id": None,
        "pain_class": None,
        "trial_id": None,
        "segment_id": None,
        "modality": None,
        "coefficient": None,
        "coefficient_type": None,
        "coefficient_level": None,
    }


    # ======================================================
    # SUBJECT ID
    # ======================================================

    subject_match = re.match(
        r"(.+?)-(BL1|PA[1-4])-",
        stem,
        flags=re.IGNORECASE,
    )

    if subject_match:

        metadata[
            "subject_id"
        ] = (
            subject_match
            .group(1)
        )


    # ======================================================
    # PAIN CLASS
    # ======================================================

    class_match = re.search(
        r"-(BL1|PA[1-4])-",
        stem,
        flags=re.IGNORECASE,
    )

    if class_match:

        metadata[
            "pain_class"
        ] = (
            class_match
            .group(1)
            .upper()
        )


    # ======================================================
    # TRIAL ID
    # ======================================================

    trial_match = re.search(
        r"-(?:BL1|PA[1-4])-(\d+)",
        stem,
        flags=re.IGNORECASE,
    )

    if trial_match:

        metadata[
            "trial_id"
        ] = (
            trial_match
            .group(1)
        )


    # ======================================================
    # SEGMENT ID
    # ======================================================

    segment_match = re.search(
        r"(?:segment|seg)[_-]?(\d+)",
        stem,
        flags=re.IGNORECASE,
    )

    if segment_match:

        metadata[
            "segment_id"
        ] = int(
            segment_match
            .group(1)
        )


    # ======================================================
    # MODALITY
    # ======================================================

    lower_stem = (
        stem.lower()
    )

    if (
        "emg_trapezius"
        in lower_stem
    ):

        metadata[
            "modality"
        ] = "emg_trapezius"

    elif "ecg" in lower_stem:

        metadata[
            "modality"
        ] = "ecg"

    elif (
        "gsr" in lower_stem
        or "eda" in lower_stem
    ):

        metadata[
            "modality"
        ] = "gsr"


    # ======================================================
    # DWT COEFFICIENT
    # ======================================================

    coefficient_match = re.search(
        r"_(A|D)(\d+)$",
        stem,
        flags=re.IGNORECASE,
    )

    if coefficient_match:

        coefficient_type = (
            coefficient_match
            .group(1)
            .upper()
        )

        coefficient_level = int(
            coefficient_match
            .group(2)
        )

        metadata[
            "coefficient_type"
        ] = coefficient_type

        metadata[
            "coefficient_level"
        ] = coefficient_level

        metadata[
            "coefficient"
        ] = (
            f"{coefficient_type}"
            f"{coefficient_level}"
        )

    return metadata


# ==========================================================
# SAMPLE KEY
# ==========================================================

def create_sample_key(
    subject_id: str,
    pain_class: str,
    trial_id: Union[
        str,
        int,
    ],
    segment_id: int,
) -> str:
    """
    Create the canonical unique BioVid segment identifier.

    Format
    ------
    subject__class__trial__segment_XXX

    Example
    -------
    071309_w_21__BL1__081__segment_000
    """

    if subject_id is None:

        raise ValueError(
            "subject_id is required."
        )

    if trial_id is None:

        raise ValueError(
            "trial_id is required."
        )

    if segment_id is None:

        raise ValueError(
            "segment_id is required."
        )

    pain_class = (
        canonicalize_class_name(
            pain_class
        )
    )

    return (
        f"{str(subject_id).strip()}__"
        f"{pain_class}__"
        f"{str(trial_id).strip()}__"
        f"segment_{int(segment_id):03d}"
    )


# ==========================================================
# SAMPLE KEY FROM METADATA
# ==========================================================

def create_sample_key_from_metadata(
    metadata: Mapping,
) -> str:
    """
    Create a sample key from a metadata dictionary.
    """

    required_keys = (
        "subject_id",
        "pain_class",
        "trial_id",
        "segment_id",
    )

    missing_keys = [
        key
        for key in required_keys
        if (
            key not in metadata
            or metadata[
                key
            ] is None
        )
    ]

    if missing_keys:

        raise ValueError(
            "Cannot construct sample_key. "
            f"Missing metadata: {missing_keys}"
        )

    return create_sample_key(
        subject_id=
            metadata[
                "subject_id"
            ],

        pain_class=
            metadata[
                "pain_class"
            ],

        trial_id=
            metadata[
                "trial_id"
            ],

        segment_id=
            metadata[
                "segment_id"
            ],
    )


# ==========================================================
# COMPLETE FILENAME METADATA
# ==========================================================

def extract_biovid_metadata(
    filename: Union[
        str,
        Path,
    ],
    require_complete: bool = True,
) -> Dict:
    """
    Parse BioVid filename and add canonical label/sample key.

    Parameters
    ----------
    filename : str or pathlib.Path
        Input filename.

    require_complete : bool
        If True, subject, class, trial and segment identifiers
        must all be recoverable.

    Returns
    -------
    dict
        Complete BioVid metadata.
    """

    metadata = (
        parse_biovid_filename(
            filename
        )
    )

    required = (
        "subject_id",
        "pain_class",
        "trial_id",
        "segment_id",
    )

    missing = [
        key
        for key in required
        if metadata[
            key
        ] is None
    ]

    if (
        require_complete
        and missing
    ):

        raise ValueError(
            f"Unable to parse complete BioVid metadata "
            f"from '{Path(filename).name}'. "
            f"Missing: {missing}"
        )

    if metadata[
        "pain_class"
    ] is not None:

        metadata[
            "label"
        ] = (
            BIOVID_CLASS_MAPPING[
                metadata[
                    "pain_class"
                ]
            ]
        )

    else:

        metadata[
            "label"
        ] = None

    if not missing:

        metadata[
            "sample_key"
        ] = (
            create_sample_key_from_metadata(
                metadata
            )
        )

    else:

        metadata[
            "sample_key"
        ] = None

    return metadata


# ==========================================================
# DATAFRAME METADATA VALIDATION
# ==========================================================

def validate_biovid_metadata_dataframe(
    dataframe: pd.DataFrame,
    require_sample_key: bool = True,
    require_subject_id: bool = True,
    require_pain_class: bool = True,
) -> pd.DataFrame:
    """
    Validate BioVid metadata contained in a DataFrame.

    Returns a cleaned copy with canonical class names and labels.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if dataframe.empty:

        raise ValueError(
            "BioVid metadata DataFrame is empty."
        )

    output = (
        dataframe.copy()
    )


    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================

    required_columns = []

    if require_sample_key:
        required_columns.append(
            "sample_key"
        )

    if require_subject_id:
        required_columns.append(
            "subject_id"
        )

    if require_pain_class:
        required_columns.append(
            "pain_class"
        )

    missing = [
        column
        for column in required_columns
        if column not in output.columns
    ]

    if missing:

        raise ValueError(
            "Missing BioVid metadata columns: "
            f"{missing}"
        )


    # ======================================================
    # SUBJECT ID
    # ======================================================

    if "subject_id" in output.columns:

        if output[
            "subject_id"
        ].isna().any():

            raise ValueError(
                "Missing subject_id values detected."
            )

        output[
            "subject_id"
        ] = (
            output[
                "subject_id"
            ]
            .astype(str)
            .str.strip()
        )


    # ======================================================
    # PAIN CLASS
    # ======================================================

    if "pain_class" in output.columns:

        if output[
            "pain_class"
        ].isna().any():

            raise ValueError(
                "Missing pain_class values detected."
            )

        output[
            "pain_class"
        ] = [
            canonicalize_class_name(
                value
            )
            for value
            in output[
                "pain_class"
            ]
        ]

        output[
            "label"
        ] = [
            BIOVID_CLASS_MAPPING[
                pain_class
            ]
            for pain_class
            in output[
                "pain_class"
            ]
        ]


    # ======================================================
    # SAMPLE KEY DUPLICATES
    # ======================================================

    if (
        "sample_key"
        in output.columns
    ):

        if output[
            "sample_key"
        ].isna().any():

            raise ValueError(
                "Missing sample_key values detected."
            )

        duplicate_mask = (
            output[
                "sample_key"
            ]
            .duplicated()
        )

        if duplicate_mask.any():

            duplicate_examples = (
                output.loc[
                    duplicate_mask,
                    "sample_key",
                ]
                .head(10)
                .tolist()
            )

            raise ValueError(
                "Duplicate BioVid sample keys detected. "
                f"Examples: {duplicate_examples}"
            )

    return output


# ==========================================================
# METADATA ALIGNMENT
# ==========================================================

def validate_biovid_alignment(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    columns: tuple[str, ...] = (
        "sample_key",
        "subject_id",
        "pain_class",
    ),
) -> None:
    """
    Verify sample alignment between BioVid modality DataFrames.

    Useful before ECG/EMG/GSR feature fusion.
    """

    if len(
        reference
    ) != len(
        other
    ):

        raise ValueError(
            "BioVid modality sample counts differ."
        )

    for column in columns:

        if (
            column not in reference.columns
            or column not in other.columns
        ):

            continue

        reference_values = (
            reference[
                column
            ]
            .astype(str)
            .to_numpy()
        )

        other_values = (
            other[
                column
            ]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            reference_values,
            other_values,
        ):

            raise ValueError(
                f"BioVid cross-modal alignment "
                f"mismatch in column '{column}'."
            )


# ==========================================================
# NUMERIC FEATURE COLUMNS
# ==========================================================

def get_biovid_feature_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return numerical feature columns excluding metadata.
    """

    feature_columns = []

    for column in (
        dataframe.columns
    ):

        if column in (
            BIOVID_METADATA_COLUMNS
        ):

            continue

        if pd.api.types.is_numeric_dtype(
            dataframe[
                column
            ]
        ):

            feature_columns.append(
                column
            )

    return feature_columns
