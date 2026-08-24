"""
Hybrid ANOVA and Random Forest feature selection.

This module implements Stage 2 of the dual-stage feature-selection
procedure used in the multimodal physiological pain-recognition
framework.

Stage 2
-------
1. Calculate univariate ANOVA F-scores.
2. Calculate Random Forest feature importances.
3. Normalize both importance measures.
4. Combine them using configurable weights.
5. Rank features by the hybrid score.
6. Retain the highest-ranked subset.

The selector must be fitted using training data only. The learned
feature subset is subsequently applied unchanged to validation
and test data.

Author
------
Shreya
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.feature_selection import (
    f_classif,
)


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_ANOVA_WEIGHT = 0.50

DEFAULT_RF_WEIGHT = 0.50

DEFAULT_SELECTION_RATIO = 0.30

DEFAULT_RANDOM_STATE = 42

DEFAULT_RF_ESTIMATORS = 300


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def _validate_inputs(
    X: pd.DataFrame,
    y,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Validate training features and labels.
    """

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "X must be a pandas DataFrame."
        )

    if X.empty:
        raise ValueError(
            "Feature matrix cannot be empty."
        )

    non_numeric_columns = [
        column
        for column in X.columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if non_numeric_columns:
        raise ValueError(
            "Feature matrix contains non-numeric columns: "
            f"{non_numeric_columns}"
        )

    X_values = X.to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(X_values)
    ):
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    y_array = np.asarray(
        y
    ).reshape(-1)

    if len(y_array) != len(X):
        raise ValueError(
            "X and y contain different numbers of samples."
        )

    if len(
        np.unique(y_array)
    ) < 2:
        raise ValueError(
            "At least two classes are required."
        )

    return (
        X,
        y_array,
    )


# ==========================================================
# SCORE NORMALIZATION
# ==========================================================

def normalize_scores(
    scores,
) -> np.ndarray:
    """
    Min-Max normalize feature scores to [0, 1].

    Equal-valued score vectors are mapped to zeros because
    they provide no ranking information.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    scores = np.nan_to_num(
        scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(
        np.min(scores)
    )

    maximum = float(
        np.max(scores)
    )

    if np.isclose(
        minimum,
        maximum,
    ):

        return np.zeros_like(
            scores
        )

    return (
        (scores - minimum)
        / (maximum - minimum)
    )


# ==========================================================
# ANOVA SCORES
# ==========================================================

def calculate_anova_scores(
    X: pd.DataFrame,
    y,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate ANOVA F-scores and p-values.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix.

    y : array-like
        Class labels.

    Returns
    -------
    f_scores : numpy.ndarray
        ANOVA F-statistics.

    p_values : numpy.ndarray
        Associated p-values.
    """

    X, y = _validate_inputs(
        X,
        y,
    )

    f_scores, p_values = (
        f_classif(
            X,
            y,
        )
    )

    f_scores = np.nan_to_num(
        f_scores,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    p_values = np.nan_to_num(
        p_values,
        nan=1.0,
        posinf=1.0,
        neginf=1.0,
    )

    return (
        f_scores.astype(
            np.float64
        ),
        p_values.astype(
            np.float64
        ),
    )


# ==========================================================
# RANDOM FOREST IMPORTANCE
# ==========================================================

def calculate_rf_importance(
    X: pd.DataFrame,
    y,
    n_estimators: int = DEFAULT_RF_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
) -> tuple[np.ndarray, RandomForestClassifier]:
    """
    Calculate Random Forest feature importance.
    """

    X, y = _validate_inputs(
        X,
        y,
    )

    if n_estimators < 1:
        raise ValueError(
            "n_estimators must be at least 1."
        )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(
        X,
        y,
    )

    importance = (
        model.feature_importances_
        .astype(
            np.float64
        )
    )

    return (
        importance,
        model,
    )


# ==========================================================
# HYBRID FEATURE RANKING
# ==========================================================

def rank_features(
    X: pd.DataFrame,
    y,
    anova_weight: float = DEFAULT_ANOVA_WEIGHT,
    rf_weight: float = DEFAULT_RF_WEIGHT,
    n_estimators: int = DEFAULT_RF_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, RandomForestClassifier]:
    """
    Rank features using combined ANOVA and RF importance.

    Hybrid score
    ------------
    hybrid_score =
        anova_weight * normalized_ANOVA
        +
        rf_weight * normalized_RF
    """

    X, y = _validate_inputs(
        X,
        y,
    )

    if anova_weight < 0:
        raise ValueError(
            "ANOVA weight cannot be negative."
        )

    if rf_weight < 0:
        raise ValueError(
            "RF weight cannot be negative."
        )

    total_weight = (
        anova_weight
        + rf_weight
    )

    if total_weight <= 0:
        raise ValueError(
            "At least one feature-selection weight "
            "must be greater than zero."
        )

    # Normalize the weights so they sum to 1.
    anova_weight = (
        anova_weight
        / total_weight
    )

    rf_weight = (
        rf_weight
        / total_weight
    )

    # ------------------------------------------------------
    # ANOVA
    # ------------------------------------------------------

    (
        anova_scores,
        p_values,
    ) = calculate_anova_scores(
        X,
        y,
    )

    normalized_anova = (
        normalize_scores(
            anova_scores
        )
    )

    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    (
        rf_importance,
        rf_model,
    ) = calculate_rf_importance(
        X=X,
        y=y,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    normalized_rf = (
        normalize_scores(
            rf_importance
        )
    )

    # ------------------------------------------------------
    # Hybrid score
    # ------------------------------------------------------

    hybrid_score = (
        anova_weight
        * normalized_anova
        +
        rf_weight
        * normalized_rf
    )

    ranking = pd.DataFrame(
        {
            "feature":
                X.columns,

            "anova_f_score":
                anova_scores,

            "anova_p_value":
                p_values,

            "anova_normalized":
                normalized_anova,

            "rf_importance":
                rf_importance,

            "rf_normalized":
                normalized_rf,

            "hybrid_score":
                hybrid_score,
        }
    )

    # Deterministic tie-breaking:
    # hybrid -> ANOVA -> RF -> feature name
    ranking = (
        ranking
        .sort_values(
            by=[
                "hybrid_score",
                "anova_normalized",
                "rf_normalized",
                "feature",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranking["rank"] = (
        np.arange(
            1,
            len(ranking) + 1,
        )
    )

    return (
        ranking,
        rf_model,
    )


# ==========================================================
# COMPLETE STAGE-2 SELECTOR
# ==========================================================

class ANOVARFFeatureSelector:
    """
    Hybrid ANOVA + Random Forest feature selector.

    Parameters
    ----------
    selection_ratio : float
        Fraction of Stage-1 features retained.

    anova_weight : float
        Contribution of normalized ANOVA score.

    rf_weight : float
        Contribution of normalized RF importance.

    n_estimators : int
        Number of Random Forest trees.

    random_state : int
        Random seed for reproducibility.

    n_jobs : int
        Number of parallel RF workers.
    """

    def __init__(
        self,
        selection_ratio: float = DEFAULT_SELECTION_RATIO,
        anova_weight: float = DEFAULT_ANOVA_WEIGHT,
        rf_weight: float = DEFAULT_RF_WEIGHT,
        n_estimators: int = DEFAULT_RF_ESTIMATORS,
        random_state: int = DEFAULT_RANDOM_STATE,
        n_jobs: int = -1,
    ):

        self.selection_ratio = (
            selection_ratio
        )

        self.anova_weight = (
            anova_weight
        )

        self.rf_weight = (
            rf_weight
        )

        self.n_estimators = (
            n_estimators
        )

        self.random_state = (
            random_state
        )

        self.n_jobs = (
            n_jobs
        )

        self.feature_names_in_ = None

        self.ranking_ = None

        self.selected_features_ = None

        self.selected_features_ranked_ = None

        self.rf_model_ = None


    def fit(
        self,
        X: pd.DataFrame,
        y,
    ) -> "ANOVARFFeatureSelector":
        """
        Fit Stage-2 feature selection using training data.
        """

        X, y = _validate_inputs(
            X,
            y,
        )

        if not (
            0 < self.selection_ratio <= 1
        ):
            raise ValueError(
                "selection_ratio must satisfy "
                "0 < selection_ratio <= 1."
            )

        self.feature_names_in_ = (
            X.columns.tolist()
        )

        (
            ranking,
            rf_model,
        ) = rank_features(
            X=X,
            y=y,
            anova_weight=
                self.anova_weight,
            rf_weight=
                self.rf_weight,
            n_estimators=
                self.n_estimators,
            random_state=
                self.random_state,
            n_jobs=
                self.n_jobs,
        )

        number_to_select = max(
            1,
            int(
                np.ceil(
                    len(
                        self.feature_names_in_
                    )
                    * self.selection_ratio
                )
            ),
        )

        selected_ranked = (
            ranking
            .head(
                number_to_select
            )[
                "feature"
            ]
            .tolist()
        )

        selected_set = set(
            selected_ranked
        )

        # Preserve original feature-column order
        # when producing model matrices.
        selected_original_order = [
            feature
            for feature
            in self.feature_names_in_
            if feature in selected_set
        ]

        ranking[
            "selected"
        ] = (
            ranking[
                "feature"
            ]
            .isin(
                selected_set
            )
        )

        self.ranking_ = (
            ranking
        )

        self.selected_features_ranked_ = (
            selected_ranked
        )

        self.selected_features_ = (
            selected_original_order
        )

        self.rf_model_ = (
            rf_model
        )

        return self


    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the fitted feature subset.
        """

        if self.selected_features_ is None:
            raise RuntimeError(
                "ANOVARFFeatureSelector must be fitted "
                "before transform()."
            )

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        missing_features = [
            feature
            for feature
            in self.selected_features_
            if feature not in X.columns
        ]

        if missing_features:
            raise ValueError(
                "Input matrix is missing selected features: "
                f"{missing_features}"
            )

        return X[
            self.selected_features_
        ].copy()


    def fit_transform(
        self,
        X: pd.DataFrame,
        y,
    ) -> pd.DataFrame:
        """
        Fit Stage-2 selector and transform training data.
        """

        self.fit(
            X,
            y,
        )

        return self.transform(
            X
        )


    def get_feature_names_out(
        self,
    ) -> List[str]:
        """
        Return selected features in model-matrix order.
        """

        if self.selected_features_ is None:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return list(
            self.selected_features_
        )


    def get_ranked_features(
        self,
    ) -> List[str]:
        """
        Return selected features from highest to lowest score.
        """

        if self.selected_features_ranked_ is None:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return list(
            self.selected_features_ranked_
        )


    def get_report(
        self,
    ) -> pd.DataFrame:
        """
        Return the complete Stage-2 ranking report.
        """

        if self.ranking_ is None:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return self.ranking_.copy()
