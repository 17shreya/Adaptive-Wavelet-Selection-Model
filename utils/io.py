"""
Input/output utilities.

This module provides standardized helpers for reading and writing
CSV, JSON, YAML, and NumPy files used by the physiological
pain-recognition pipeline.

Scientific modules should return Python objects/DataFrames rather
than writing files directly. Experiment scripts may use these
functions to persist results.

Author
------
Shreya
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import pandas as pd
import yaml


PathLike = Union[
    str,
    Path,
]


# ==========================================================
# PATH PREPARATION
# ==========================================================

def _prepare_parent_directory(
    path: PathLike,
) -> Path:
    """
    Resolve a file path and create its parent directory.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


# ==========================================================
# CSV
# ==========================================================

def read_csv(
    path: PathLike,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a CSV file with basic validation.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    dataframe = pd.read_csv(
        path,
        **kwargs,
    )

    return dataframe


def write_csv(
    dataframe: pd.DataFrame,
    path: PathLike,
    index: bool = False,
    **kwargs,
) -> Path:
    """
    Save a DataFrame as CSV.

    Returns
    -------
    pathlib.Path
        Saved file path.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        raise TypeError(
            "write_csv expects a pandas DataFrame."
        )

    path = _prepare_parent_directory(
        path
    )

    dataframe.to_csv(
        path,
        index=index,
        **kwargs,
    )

    return path


# ==========================================================
# YAML
# ==========================================================

def read_yaml(
    path: PathLike,
) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"YAML file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        configuration = yaml.safe_load(
            file
        )

    if configuration is None:

        configuration = {}

    if not isinstance(
        configuration,
        dict,
    ):

        raise ValueError(
            "Top-level YAML content must be a mapping."
        )

    return configuration


def write_yaml(
    data: Mapping,
    path: PathLike,
) -> Path:
    """
    Save a mapping as YAML.
    """

    path = _prepare_parent_directory(
        path
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        yaml.safe_dump(
            dict(data),
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    return path


# ==========================================================
# JSON
# ==========================================================

def read_json(
    path: PathLike,
) -> Any:
    """
    Read a JSON file.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


def write_json(
    data: Any,
    path: PathLike,
    indent: int = 4,
) -> Path:
    """
    Save JSON-serializable data.

    NumPy scalars and arrays are converted automatically.
    """

    path = _prepare_parent_directory(
        path
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=indent,
            ensure_ascii=False,
            default=_json_converter,
        )

    return path


def _json_converter(
    value,
):
    """
    Convert common NumPy objects to JSON-compatible types.
    """

    if isinstance(
        value,
        np.integer,
    ):

        return int(
            value
        )

    if isinstance(
        value,
        np.floating,
    ):

        return float(
            value
        )

    if isinstance(
        value,
        np.ndarray,
    ):

        return value.tolist()

    if isinstance(
        value,
        Path,
    ):

        return str(
            value
        )

    raise TypeError(
        f"Object of type "
        f"{type(value).__name__} "
        f"is not JSON serializable."
    )


# ==========================================================
# NUMPY
# ==========================================================

def save_numpy(
    array,
    path: PathLike,
) -> Path:
    """
    Save a NumPy array in .npy format.
    """

    path = _prepare_parent_directory(
        path
    )

    np.save(
        path,
        np.asarray(
            array
        ),
    )

    return path


def load_numpy(
    path: PathLike,
) -> np.ndarray:
    """
    Load a NumPy .npy file.
    """

    path = Path(
        path
    )

    if not path.exists():

        raise FileNotFoundError(
            f"NumPy file not found: {path}"
        )

    return np.load(
        path,
        allow_pickle=False,
    )


# ==========================================================
# CONFIGURATION LOADER
# ==========================================================

def load_config(
    path: PathLike,
) -> Dict[str, Any]:
    """
    Load repository YAML configuration.

    This is an alias around ``read_yaml`` intended for
    experiment scripts.
    """

    return read_yaml(
        path
    )
