"""
Cross-platform repository path utilities.

This module provides repository-relative paths without relying on
machine-specific Windows or Linux directory names.

No dataset path is hard-coded into the source code.

Author
------
Shreya
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[
    str,
    Path,
]


# ==========================================================
# REPOSITORY ROOT
# ==========================================================

def get_project_root() -> Path:
    """
    Return the root directory of the repository.

    Expected layout
    ---------------
    repository/
        configs/
        data/
        src/
        scripts/
        results/
    """

    current_file = Path(
        __file__
    ).resolve()

    # paths.py
    # -> utils
    # -> pain_recognition
    # -> src
    # -> repository root

    return current_file.parents[
        3
    ]


# ==========================================================
# STANDARD REPOSITORY DIRECTORIES
# ==========================================================

def get_config_dir() -> Path:
    """
    Return configs directory.
    """

    return (
        get_project_root()
        / "configs"
    )


def get_data_dir() -> Path:
    """
    Return data directory.
    """

    return (
        get_project_root()
        / "data"
    )


def get_raw_data_dir() -> Path:
    """
    Return raw-data directory.
    """

    return (
        get_data_dir()
        / "raw"
    )


def get_interim_data_dir() -> Path:
    """
    Return interim-data directory.
    """

    return (
        get_data_dir()
        / "interim"
    )


def get_processed_data_dir() -> Path:
    """
    Return processed-data directory.
    """

    return (
        get_data_dir()
        / "processed"
    )


def get_results_dir() -> Path:
    """
    Return results directory.
    """

    return (
        get_project_root()
        / "results"
    )


def get_figures_dir() -> Path:
    """
    Return figures directory.
    """

    return (
        get_project_root()
        / "figures"
    )


def get_docs_dir() -> Path:
    """
    Return documentation directory.
    """

    return (
        get_project_root()
        / "docs"
    )


# ==========================================================
# CONFIGURATION PATH
# ==========================================================

def get_config_path(
    filename: str,
) -> Path:
    """
    Return path to one configuration file.

    Example
    -------
    get_config_path("biovid.yaml")
    """

    if not filename:
        raise ValueError(
            "Configuration filename cannot be empty."
        )

    return (
        get_config_dir()
        / filename
    )


# ==========================================================
# RESULT PATH
# ==========================================================

def get_result_path(
    *parts: str,
    create_parent: bool = False,
) -> Path:
    """
    Construct a path inside the results directory.

    Example
    -------
    get_result_path(
        "biovid",
        "binary",
        "summary.csv"
    )
    """

    path = (
        get_results_dir()
        .joinpath(
            *parts
        )
    )

    if create_parent:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    return path


# ==========================================================
# FIGURE PATH
# ==========================================================

def get_figure_path(
    *parts: str,
    create_parent: bool = False,
) -> Path:
    """
    Construct a path inside the figures directory.
    """

    path = (
        get_figures_dir()
        .joinpath(
            *parts
        )
    )

    if create_parent:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    return path


# ==========================================================
# USER-SUPPLIED DATASET ROOT
# ==========================================================

def resolve_dataset_root(
    dataset_root: PathLike,
    must_exist: bool = True,
) -> Path:
    """
    Resolve a user-supplied dataset directory.

    Raw BioVid/X-ITE datasets are not included in this repository,
    therefore dataset locations must be supplied externally.

    Parameters
    ----------
    dataset_root : str or pathlib.Path
        User-specified dataset location.

    must_exist : bool, optional
        Require directory to exist.

    Returns
    -------
    pathlib.Path
        Absolute resolved dataset path.
    """

    path = (
        Path(
            dataset_root
        )
        .expanduser()
        .resolve()
    )

    if (
        must_exist
        and not path.exists()
    ):

        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{path}"
        )

    if (
        must_exist
        and not path.is_dir()
    ):

        raise NotADirectoryError(
            f"Dataset root is not a directory: "
            f"{path}"
        )

    return path


# ==========================================================
# ENSURE DIRECTORY
# ==========================================================

def ensure_directory(
    path: PathLike,
) -> Path:
    """
    Create a directory if it does not already exist.
    """

    directory = Path(
        path
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory
