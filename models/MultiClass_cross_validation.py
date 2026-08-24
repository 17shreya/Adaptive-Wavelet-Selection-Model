"""
Subject-independent cross-validation.

This module implements Stratified Group K-Fold evaluation for
physiological pain-recognition experiments.

The grouping variable should normally be the subject/participant ID,
ensuring that samples from one subject cannot occur in both training
and test partitions of the same fold.

Reported metrics
----------------
Accuracy
Balanced Accuracy
Macro Precision
Macro Recall
Macro F1
Weighted F1
Cohen's Kappa
Macro ROC-AUC OvR

The module returns results and predictions but performs no file
saving or plotting.

Author
------
Shreya
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from sklearn.model_selection import (
    StratifiedGroupKFold,
)


# ==========================================================
# DEFAULT CONFIGURATION
# ==========================================================

DEFAULT_N_SPLITS = 5

DEFAULT_RANDOM_STATE = 42


# ==========================================================
# INPUT VALIDATION
# ==========================================================

def validate_cv_inputs(
    X,
    y,
    groups,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Validate feature matrix, labels and grouping variable.
    """

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    y = np.asarray(
        y
    ).reshape(-1)

    groups = np.asarray(
        groups
    ).astype(str).reshape(-1)

    if X.ndim != 2:

        raise ValueError(
            "X must be a two-dimensional feature matrix."
        )

    if not (
        X.shape[0]
        == len(y)
        == len(groups)
    ):

        raise ValueError(
            "X, y and groups must contain "
            "the same number of samples."
        )

    if X.shape[1] == 0:

        raise ValueError(
            "Feature matrix contains no features."
        )

    if not np.all(
        np.isfinite(X)
    ):

        raise ValueError(
            "Feature matrix contains NaN "
            "or infinite values."
        )

    if len(
        np.unique(y)
    ) < 2:

        raise ValueError(
            "At least two classes are required."
        )

    if len(
        np.unique(groups)
    ) < 2:

        raise ValueError(
            "At least two subject groups are required."
        )

    return (
        X,
        y,
        groups,
    )


# ==========================================================
# CREATE CV SPLITTER
# ==========================================================

def create_stratified_group_cv(
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    shuffle: bool = True,
) -> StratifiedGroupKFold:
    """
    Create the subject-independent CV splitter.
    """

    if n_splits < 2:

        raise ValueError(
            "n_splits must be at least 2."
        )

    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=(
            random_state
            if shuffle
            else None
        ),
    )


# ==========================================================
# GENERATE FOLD INDICES
# ==========================================================

def generate_cv_splits(
    y,
    groups,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Generate subject-independent train/test indices.

    This function is useful when preprocessing, feature selection,
    and fusion must be fitted separately inside each fold.

    Yields
    ------
    fold_number : int

    train_indices : numpy.ndarray

    test_indices : numpy.ndarray
    """

    y = np.asarray(
        y
    ).reshape(-1)

    groups = np.asarray(
        groups
    ).astype(str).reshape(-1)

    if len(y) != len(groups):

        raise ValueError(
            "y and groups must contain "
            "the same number of samples."
        )

    cv = create_stratified_group_cv(
        n_splits=n_splits,
        random_state=random_state,
    )

    dummy_X = np.zeros(
        (
            len(y),
            1,
        ),
        dtype=np.float64,
    )

    for fold, (
        train_indices,
        test_indices,
    ) in enumerate(
        cv.split(
            dummy_X,
            y,
            groups,
        ),
        start=1,
    ):

        audit_subject_leakage(
            groups_train=
                groups[
                    train_indices
                ],

            groups_test=
                groups[
                    test_indices
                ],

            fold=fold,
        )

        yield (
            fold,
            train_indices,
            test_indices,
        )


# ==========================================================
# SUBJECT LEAKAGE AUDIT
# ==========================================================

def audit_subject_leakage(
    groups_train,
    groups_test,
    fold: Optional[int] = None,
) -> None:
    """
    Verify that training and test subjects do not overlap.
    """

    train_subjects = set(
        np.asarray(
            groups_train
        ).astype(str)
    )

    test_subjects = set(
        np.asarray(
            groups_test
        ).astype(str)
    )

    overlap = (
        train_subjects
        & test_subjects
    )

    if overlap:

        fold_text = (
            f" in fold {fold}"
            if fold is not None
            else ""
        )

        raise RuntimeError(
            "Subject leakage detected"
            f"{fold_text}: "
            f"{sorted(overlap)}"
        )


# ==========================================================
# MULTICLASS ROC-AUC
# ==========================================================

def calculate_macro_roc_auc(
    model: BaseEstimator,
    X_test,
    y_test,
    class_labels: Sequence,
) -> float:
    """
    Calculate macro one-vs-rest ROC-AUC.

    NaN is returned when a test fold does not contain all expected
    classes or probability estimates are unavailable.
    """

    y_test = np.asarray(
        y_test
    ).reshape(-1)

    test_classes = set(
        np.unique(
            y_test
        )
    )

    expected_classes = set(
        class_labels
    )

    if len(
        test_classes
    ) < 2:

        return np.nan

    if test_classes != expected_classes:

        return np.nan

    if not hasattr(
        model,
        "predict_proba",
    ):

        return np.nan

    try:

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        model_classes = np.asarray(
            model.classes_
        )

        if set(
            model_classes
        ) != expected_classes:

            return np.nan

        return float(
            roc_auc_score(
                y_test,
                probabilities,
                labels=list(
                    class_labels
                ),
                multi_class="ovr",
                average="macro",
            )
        )

    except Exception:

        return np.nan


# ==========================================================
# BINARY ROC-AUC
# ==========================================================

def calculate_binary_roc_auc(
    model: BaseEstimator,
    X_test,
    y_test,
    positive_label=1,
) -> float:
    """
    Calculate binary ROC-AUC when probabilities are available.
    """

    y_test = np.asarray(
        y_test
    ).reshape(-1)

    if len(
        np.unique(
            y_test
        )
    ) < 2:

        return np.nan

    if not hasattr(
        model,
        "predict_proba",
    ):

        return np.nan

    try:

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        model_classes = np.asarray(
            model.classes_
        )

        positive_positions = np.where(
            model_classes
            == positive_label
        )[0]

        if len(
            positive_positions
        ) != 1:

            return np.nan

        positive_probability = (
            probabilities[
                :,
                positive_positions[0],
            ]
        )

        binary_truth = (
            np.asarray(
                y_test
            )
            == positive_label
        ).astype(int)

        return float(
            roc_auc_score(
                binary_truth,
                positive_probability,
            )
        )

    except Exception:

        return np.nan


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

def calculate_classification_metrics(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """
    Calculate classification performance metrics.
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)

    return {
        "Accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "Balanced_Accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "Macro_Precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "Macro_Recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "Macro_F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "Weighted_F1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                )
            ),

        "Cohen_Kappa":
            float(
                cohen_kappa_score(
                    y_true,
                    y_pred,
                )
            ),
    }


# ==========================================================
# EVALUATE MODELS ON ONE FOLD
# ==========================================================

def evaluate_models_on_fold(
    X_train,
    y_train,
    X_test,
    y_test,
    groups_train,
    groups_test,
    models: Mapping[
        str,
        BaseEstimator,
    ],
    fold: int,
    representation: str,
    class_labels: Sequence,
) -> Tuple[
    list[dict],
    Dict[str, dict],
]:
    """
    Train and evaluate all classifiers on one CV fold.

    The feature matrices supplied here should already have undergone
    any fold-specific feature selection and fusion.
    """

    X_train = np.asarray(
        X_train,
        dtype=np.float64,
    )

    X_test = np.asarray(
        X_test,
        dtype=np.float64,
    )

    y_train = np.asarray(
        y_train
    ).reshape(-1)

    y_test = np.asarray(
        y_test
    ).reshape(-1)

    groups_train = np.asarray(
        groups_train
    ).astype(str).reshape(-1)

    groups_test = np.asarray(
        groups_test
    ).astype(str).reshape(-1)

    audit_subject_leakage(
        groups_train=
            groups_train,

        groups_test=
            groups_test,

        fold=fold,
    )

    fold_records = []

    predictions = {}

    for (
        model_name,
        estimator,
    ) in models.items():

        model = clone(
            estimator
        )

        model.fit(
            X_train,
            y_train,
        )

        y_pred = model.predict(
            X_test
        )

        metrics = (
            calculate_classification_metrics(
                y_true=y_test,
                y_pred=y_pred,
            )
        )

        if len(
            class_labels
        ) == 2:

            roc_auc = (
                calculate_binary_roc_auc(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    positive_label=
                        class_labels[-1],
                )
            )

        else:

            roc_auc = (
                calculate_macro_roc_auc(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    class_labels=
                        class_labels,
                )
            )

        record = {
            "Representation":
                representation,

            "Classifier":
                model_name,

            "Fold":
                fold,

            "Train_Samples":
                len(
                    y_train
                ),

            "Test_Samples":
                len(
                    y_test
                ),

            "Train_Subjects":
                len(
                    np.unique(
                        groups_train
                    )
                ),

            "Test_Subjects":
                len(
                    np.unique(
                        groups_test
                    )
                ),

            "Subject_Overlap":
                0,

            **metrics,

            "ROC_AUC":
                roc_auc,
        }

        fold_records.append(
            record
        )

        predictions[
            model_name
        ] = {
            "y_true":
                y_test.copy(),

            "y_pred":
                np.asarray(
                    y_pred
                ).copy(),

            "groups":
                groups_test.copy(),
        }

    return (
        fold_records,
        predictions,
    )


# ==========================================================
# STANDARD CROSS-VALIDATION
# ==========================================================

def evaluate_classifiers_cv(
    X,
    y,
    groups,
    models: Mapping[
        str,
        BaseEstimator,
    ],
    representation: str = "features",
    class_labels: Optional[Sequence] = None,
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, dict],
]:
    """
    Perform complete subject-independent cross-validation.

    Important
    ---------
    Use this function only when X is already a valid fold-independent
    representation.

    If feature selection is data-driven, use ``generate_cv_splits()``
    and fit feature selection separately within each training fold.

    Returns
    -------
    fold_results : pandas.DataFrame

    summary_results : pandas.DataFrame

    pooled_predictions : dict
    """

    (
        X,
        y,
        groups,
    ) = validate_cv_inputs(
        X,
        y,
        groups,
    )

    if class_labels is None:

        class_labels = sorted(
            np.unique(
                y
            ).tolist()
        )

    fold_records = []

    pooled_predictions = {
        model_name: {
            "y_true": [],
            "y_pred": [],
            "groups": [],
        }
        for model_name
        in models
    }

    for (
        fold,
        train_indices,
        test_indices,
    ) in generate_cv_splits(
        y=y,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
    ):

        (
            current_records,
            current_predictions,
        ) = evaluate_models_on_fold(
            X_train=
                X[
                    train_indices
                ],

            y_train=
                y[
                    train_indices
                ],

            X_test=
                X[
                    test_indices
                ],

            y_test=
                y[
                    test_indices
                ],

            groups_train=
                groups[
                    train_indices
                ],

            groups_test=
                groups[
                    test_indices
                ],

            models=models,

            fold=fold,

            representation=
                representation,

            class_labels=
                class_labels,
        )

        fold_records.extend(
            current_records
        )

        for model_name in models:

            pooled_predictions[
                model_name
            ]["y_true"].extend(
                current_predictions[
                    model_name
                ]["y_true"].tolist()
            )

            pooled_predictions[
                model_name
            ]["y_pred"].extend(
                current_predictions[
                    model_name
                ]["y_pred"].tolist()
            )

            pooled_predictions[
                model_name
            ]["groups"].extend(
                current_predictions[
                    model_name
                ]["groups"].tolist()
            )

    fold_results = pd.DataFrame(
        fold_records
    )

    summary_results = (
        summarize_cv_results(
            fold_results
        )
    )

    return (
        fold_results,
        summary_results,
        pooled_predictions,
    )


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

def summarize_cv_results(
    fold_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and standard deviation across CV folds.
    """

    if fold_results.empty:

        raise ValueError(
            "Fold-result table is empty."
        )

    metric_columns = [
        "Accuracy",
        "Balanced_Accuracy",
        "Macro_Precision",
        "Macro_Recall",
        "Macro_F1",
        "Weighted_F1",
        "Cohen_Kappa",
        "ROC_AUC",
    ]

    summary_records = []

    grouped = (
        fold_results.groupby(
            [
                "Representation",
                "Classifier",
            ],
            sort=False,
        )
    )

    for (
        representation,
        classifier,
    ), subset in grouped:

        record = {
            "Representation":
                representation,

            "Classifier":
                classifier,

            "Number_of_Folds":
                len(
                    subset
                ),
        }

        for metric in metric_columns:

            values = (
                subset[
                    metric
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            finite_values = (
                values[
                    np.isfinite(
                        values
                    )
                ]
            )

            if len(
                finite_values
            ) == 0:

                mean_value = np.nan
                std_value = np.nan

            else:

                mean_value = float(
                    np.mean(
                        finite_values
                    )
                )

                std_value = (
                    float(
                        np.std(
                            finite_values,
                            ddof=1,
                        )
                    )
                    if len(
                        finite_values
                    ) > 1
                    else 0.0
                )

            record[
                f"{metric}_Mean"
            ] = mean_value

            record[
                f"{metric}_Std"
            ] = std_value

            if (
                np.isfinite(
                    mean_value
                )
                and
                np.isfinite(
                    std_value
                )
            ):

                record[
                    f"{metric}_Mean_SD"
                ] = (
                    f"{100 * mean_value:.2f}"
                    f" ± "
                    f"{100 * std_value:.2f}"
                )

            else:

                record[
                    f"{metric}_Mean_SD"
                ] = "NA"

        summary_records.append(
            record
        )

    return pd.DataFrame(
        summary_records
    )


# ==========================================================
# POOLED CONFUSION MATRIX
# ==========================================================

def calculate_pooled_confusion_matrix(
    pooled_predictions: Mapping,
    classifier: str,
    class_labels: Sequence,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Calculate pooled out-of-fold confusion matrix.

    Parameters
    ----------
    normalize : {None, "true", "pred", "all"}
        sklearn confusion-matrix normalization mode.
    """

    if classifier not in (
        pooled_predictions
    ):

        raise KeyError(
            f"Classifier '{classifier}' not found."
        )

    values = (
        pooled_predictions[
            classifier
        ]
    )

    y_true = np.asarray(
        values[
            "y_true"
        ]
    )

    y_pred = np.asarray(
        values[
            "y_pred"
        ]
    )

    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            class_labels
        ),
        normalize=normalize,
    )


# ==========================================================
# POOLED PREDICTION DATAFRAME
# ==========================================================

def pooled_predictions_to_dataframe(
    pooled_predictions: Mapping,
    classifier: str,
) -> pd.DataFrame:
    """
    Convert pooled OOF predictions to a DataFrame.
    """

    if classifier not in (
        pooled_predictions
    ):

        raise KeyError(
            f"Classifier '{classifier}' not found."
        )

    values = (
        pooled_predictions[
            classifier
        ]
    )

    return pd.DataFrame(
        {
            "subject_id":
                values[
                    "groups"
                ],

            "y_true":
                values[
                    "y_true"
                ],

            "y_pred":
                values[
                    "y_pred"
                ],
        }
    )


from pain_recognition.models.classifiers import (
    create_classifiers,
)

from pain_recognition.evaluation.cross_validation import (
    evaluate_classifiers_cv,
)


models = create_classifiers(
    random_state=42
)


fold_results, summary_results, pooled_predictions = (
    evaluate_classifiers_cv(
        X=X_fusion,
        y=y,
        groups=subject_ids,
        models=models,
        representation="ECG + EMG + GSR",
        class_labels=[0, 1, 2, 3, 4],
        n_splits=5,
        random_state=42,
    )
)


all_fold_results = []
all_summary_results = []

for representation_name, X_representation in (
    representations.items()
):

    fold_df, summary_df, pooled = (
        evaluate_classifiers_cv(
            X=X_representation,
            y=y,
            groups=subject_ids,
            models=create_classifiers(
                random_state=42
            ),
            representation=
                representation_name,
            class_labels=[
                0,
                1,
                2,
                3,
                4,
            ],
            n_splits=5,
            random_state=42,
        )
    )

    all_fold_results.append(
        fold_df
    )

    all_summary_results.append(
        summary_df
    )


all_fold_results = pd.concat(
    all_fold_results,
    ignore_index=True,
)

all_summary_results = pd.concat(
    all_summary_results,
    ignore_index=True,
)
