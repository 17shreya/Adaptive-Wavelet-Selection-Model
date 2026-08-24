"""
Reproducibility utilities.

This module centralizes random-seed configuration used throughout
the physiological pain-recognition experiments.

The repository primarily uses NumPy and scikit-learn. Scikit-learn
estimators should additionally receive ``random_state`` explicitly
when instantiated.

Author
------
Shreya
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np


# ==========================================================
# DEFAULT RANDOM SEED
# ==========================================================

DEFAULT_RANDOM_STATE = 42


# ==========================================================
# SET GLOBAL SEEDS
# ==========================================================

def set_global_seed(
    seed: int = DEFAULT_RANDOM_STATE,
) -> int:
    """
    Set reproducibility seeds for Python and NumPy.

    Parameters
    ----------
    seed : int, optional
        Random seed.

    Returns
    -------
    int
        Seed that was applied.
    """

    if not isinstance(
        seed,
        int,
    ):
        raise TypeError(
            "Seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "Seed must be non-negative."
        )

    os.environ[
        "PYTHONHASHSEED"
    ] = str(
        seed
    )

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    return seed


# ==========================================================
# REPRODUCIBILITY CONFIGURATION
# ==========================================================

def get_reproducibility_settings(
    seed: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, int]:
    """
    Return reproducibility settings for reporting/configuration.
    """

    return {
        "python_random_seed":
            int(seed),

        "numpy_seed":
            int(seed),

        "sklearn_random_state":
            int(seed),
    }


from pain_recognition.utils.seeds import (
    set_global_seed,
)

set_global_seed(42)


