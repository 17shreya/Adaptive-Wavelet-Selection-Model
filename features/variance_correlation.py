"""
Variance and correlation-based feature filtering.

This module implements Stage 1 of the dual-stage feature-selection
procedure used in the multimodal physiological pain-recognition
framework.

Stage 1
-------
1. Remove features with variance below a specified threshold.
2. Remove highly correlated redundant features.

The selector must be fitted using training data only. The learned
feature subset is then applied unchanged to validation and test data.

No dataset paths, file saving, or experiment-specific processing
are implemented in this module.

Author
------
Shreya
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ==========================================================
# DEFAULT PARAMETERS
# ==========================================================

DEFAULT_VARIANCE_THRESHOLD = 0.0

DEFAULT_CORRELATION_THRESHOLD = 0.95


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def _validate_feature_dataframe(
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a numerical feature matrix.

    Parameters
    ----------
    X : pandas.DataFrame
        Input feature matrix.

    Returns
    -------
    pandas.DataFrame
        Validated feature matrix.

    Raises
    ------
    TypeError
        If input is not a DataFrame.

    ValueError
        If the feature matrix is empty, contains non-numeric
        columns, or contains NaN/Inf values.
    """

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise TypeError(
            "Feature matrix must be a pandas DataFrame."
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

    values = X.to_numpy(
        dtype=np.float64
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return X


# ==========================================================
# VARIANCE FILTERING
# ==========================================================

def calculate_feature_variances(
    X: pd.DataFrame,
) -> pd.Series:
    """
    Calculate population variance for every feature.

    Parameters
    ----------
    X : pandas.DataFrame
        Numerical feature matrix.

    Returns
    -------
    pandas.Series
        Variance indexed by feature name.
    """

    X = _validate_feature_dataframe(
        X
    )

    variances = np.var(
        X.to_numpy(
            dtype=np.float64
        ),
        axis=0,
        ddof=0,
    )

    return pd.Series(
        variances,
        index=X.columns,
        name="variance",
    )


def variance_filter(
    X: pd.DataFrame,
    threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> tuple[pd.DataFrame, List[str], pd.Series]:
    """
    Remove features whose variance is not greater than the threshold.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix.
    threshold : float, optional
        Minimum required feature variance.

    Returns
    -------
    filtered_X : pandas.DataFrame
        Variance-filtered feature matrix.
    selected_features : list of str
        Features retained after variance filtering.
    variances : pandas.Series
        Variance of every original feature.
    """

    if threshold < 0:
        raise ValueError(
            "Variance threshold cannot be negative."
        )

    variances = (
        calculate_feature_variances(
            X
        )
    )

    selected_features = (
        variances[
            variances > threshold
        ]
        .index
        .tolist()
    )

    if not selected_features:
        raise ValueError(
            "Variance filtering removed all features."
        )

    filtered_X = X[
        selected_features
    ].copy()

    return (
        filtered_X,
        selected_features,
        variances,
    )


# ==========================================================
# CORRELATION FILTERING
# ==========================================================

def correlation_filter(
    X: pd.DataFrame,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
) -> tuple[pd.DataFrame, List[str], List[str], pd.DataFrame]:
    """
    Remove highly correlated redundant features.

    Pearson absolute correlation is calculated between all pairs
    of features. For each correlated pair exceeding the threshold,
    the later feature in the deterministic feature ordering is
    removed.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix after variance filtering.
    threshold : float, optional
        Maximum permitted absolute Pearson correlation.

    Returns
    -------
    filtered_X : pandas.DataFrame
        Correlation-filtered feature matrix.
    selected_features : list of str
        Retained features.
    removed_features : list of str
        Features removed because of high correlation.
    correlation_matrix : pandas.DataFrame
        Absolute Pearson correlation matrix.
    """

    X = _validate_feature_dataframe(
        X
    )

    if not 0 < threshold <= 1:
        raise ValueError(
            "Correlation threshold must satisfy "
            "0 < threshold <= 1."
        )

    correlation_matrix = (
        X.corr(
            method="pearson"
        )
        .abs()
        .fillna(0.0)
    )

    upper_triangle = (
        correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )
    )

    removed_features = [
        column
        for column in upper_triangle.columns
        if (
            upper_triangle[column]
            > threshold
        ).any()
    ]

    selected_features = [
        column
        for column in X.columns
        if column not in removed_features
    ]

    if not selected_features:
        raise ValueError(
            "Correlation filtering removed all features."
        )

    filtered_X = X[
        selected_features
    ].copy()

    return (
        filtered_X,
        selected_features,
        removed_features,
        correlation_matrix,
    )


# ==========================================================
# COMPLETE STAGE-1 SELECTOR
# ==========================================================

class VarianceCorrelationSelector:
    """
    Combined variance and correlation feature selector.

    The selector is fitted on training data and subsequently used
    to transform validation/test data using exactly the same
    feature subset.

    Parameters
    ----------
    variance_threshold : float
        Minimum required feature variance.

    correlation_threshold : float
        Maximum permitted absolute pairwise Pearson correlation.
    """

    def __init__(
        self,
        variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
        correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    ):

        self.variance_threshold = (
            variance_threshold
        )

        self.correlation_threshold = (
            correlation_threshold
        )

        self.feature_names_in_ = None

        self.variances_ = None

        self.variance_selected_features_ = None

        self.correlation_removed_features_ = None

        self.selected_features_ = None

        self.correlation_matrix_ = None


    def fit(
        self,
        X: pd.DataFrame,
    ) -> "VarianceCorrelationSelector":
        """
        Fit Stage-1 feature filtering using training data.
        """

        X = _validate_feature_dataframe(
            X
        )

        self.feature_names_in_ = (
            X.columns.tolist()
        )

        (
            variance_filtered,
            variance_features,
            variances,
        ) = variance_filter(
            X=X,
            threshold=
                self.variance_threshold,
        )

        (
            _,
            final_features,
            removed_features,
            correlation_matrix,
        ) = correlation_filter(
            X=variance_filtered,
            threshold=
                self.correlation_threshold,
        )

        self.variances_ = (
            variances
        )

        self.variance_selected_features_ = (
            variance_features
        )

        self.correlation_removed_features_ = (
            removed_features
        )

        self.selected_features_ = (
            final_features
        )

        self.correlation_matrix_ = (
            correlation_matrix
        )

        return self


    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply the fitted Stage-1 feature subset.
        """

        if self.selected_features_ is None:
            raise RuntimeError(
                "VarianceCorrelationSelector must be fitted "
                "before transform()."
            )

        X = _validate_feature_dataframe(
            X
        )

        missing_features = [
            feature
            for feature
            in self.selected_features_
            if feature not in X.columns
        ]

        if missing_features:
            raise ValueError(
                "Input matrix is missing fitted features: "
                f"{missing_features}"
            )

        return X[
            self.selected_features_
        ].copy()


    def fit_transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit selector and transform training data.
        """

        self.fit(
            X
        )

        return self.transform(
            X
        )


    def get_feature_names_out(
        self,
    ) -> List[str]:
        """
        Return selected feature names.
        """

        if self.selected_features_ is None:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        return list(
            self.selected_features_
        )


    def get_report(
        self,
    ) -> pd.DataFrame:
        """
        Generate a feature-level Stage-1 selection report.
        """

        if self.feature_names_in_ is None:
            raise RuntimeError(
                "Selector has not been fitted."
            )

        records = []

        for feature in (
            self.feature_names_in_
        ):

            variance_pass = (
                feature
                in self.variance_selected_features_
            )

            correlation_pass = (
                feature
                in self.selected_features_
            )

            records.append(
                {
                    "feature":
                        feature,

                    "variance":
                        float(
                            self.variances_[
                                feature
                            ]
                        ),

                    "variance_pass":
                        variance_pass,

                    "correlation_pass":
                        correlation_pass,

                    "selected":
                        correlation_pass,
                }
            )

        return pd.DataFrame(
            records
        )
