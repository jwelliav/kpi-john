import numpy as np
import pandas as pd
import sklearn.metrics as skm
from typing import Dict, List


TARGETS = [x / 10 for x in range(10)]


def nan_ok(func):
    def wrapper(*args):
        try:
            return float(func(*args))
        except:
            return float(np.nan)
    return wrapper


@nan_ok
def compute_precision_at_recall(
    precision: np.ndarray, recall: np.ndarray, target_recall: float
) -> float:
    return np.interp(target_recall, recall[::-1], precision[::-1])


@nan_ok
def compute_thh_at_recall(
    threshold: np.ndarray, recall: np.ndarray, target_recall: float
) -> float:
    return np.interp(target_recall, recall[:-1][::-1], threshold[::-1])


@nan_ok
def compute_recall_at_precision(
    precision: np.ndarray, recall: np.ndarray, target_precision: float
) -> float:
    return np.interp(target_precision, precision, recall)


@nan_ok
def compute_thh_at_precision(
    threshold: np.ndarray, precision: np.ndarray, target_precision: float
) -> float:
    return np.interp(target_precision, precision[:-1], threshold)


@nan_ok
def compute_cohen_kappa(y1, y2):
    return skm.cohen_kappa_score(y1, y2)


@nan_ok
def compute_rocauc(y_fact, y_pred):
    return skm.roc_auc_score(y_fact, y_pred)


@nan_ok
def compute_prauc(y_fact, y_pred):
    return skm.average_precision_score(y_fact, y_pred)


@nan_ok
def compute_f1_score(y_fact, y_pred):
    return skm.f1_score(y_fact, y_pred)



def get_metrics_from_y_values(
    df: pd.DataFrame,
    targets: List[float] = TARGETS,
    y_fact: str = "y_fact",
    y_pred: str = "y_pred",
) -> Dict[str, np.float64]:

    """Returns a dictionary of computed KPI metrics, with metric names as keys and
    computed values as values.

    :param df: A dataframe consisting of true y values, and predicted y probabilities.
    :type df: pandas.DataFrame
    :param recalls: A list of floats corresponding to the values of recall at which to
        compute various KPI metrics.
    :type recalls: List[float]
    :param y_fact: The column name for the true y values.
    :type y_fact: str, optional
    :param y_pred: The column name for the predicted y probabilities.
    :type y_pred: str, optional
    :return: A dictionary of computed KPI metrics.
    :rtype: Dict[str, numpy.float64]
    """

    y_fact = df[y_fact].astype(int)
    y_pred = df[y_pred].astype(float)
    
    precision, recall, threshold = skm.precision_recall_curve(y_fact, y_pred)

    precision_neg, recall_neg, threshold_neg = skm.precision_recall_curve(1 - y_fact, 1 - y_pred)


    metrics = dict()
    for target in targets:

        metrics[f"p@r={target}"] = compute_precision_at_recall(
            precision, recall, target
        )

        metrics[f"thh@r={target}"] = compute_thh_at_recall(
            threshold, recall, target
        )

        metrics[f"cohen_kappa_score@r={target}"] = compute_cohen_kappa(
            y_fact, (y_pred >= metrics[f"thh@r={target}"]).astype(int)
        )

        metrics[f"r@p={target}"] = compute_recall_at_precision(
            precision, recall, target
        )

        metrics[f"thh@p={target}"] = compute_thh_at_precision(
            threshold, precision, target
        )

        metrics[f"cohen_kappa_score@p={target}"] = compute_cohen_kappa(
            y_fact, (y_pred >= metrics[f"thh@p={target}"]).astype(int)
        )

        metrics[f"cohen_kappa_score@thh={target}"] = compute_cohen_kappa(
            y_fact, (y_pred >= target).astype(int)
        )

        metrics[f"f1@r={target}"] = compute_f1_score(
            y_fact, (y_pred >= metrics[f"thh@r={target}"]).astype(int)
        )

        metrics[f"negative_p@r={target}"] = compute_precision_at_recall(
            precision_neg, recall_neg, target
        )

        metrics[f"negative_thh@r={target}"] = compute_thh_at_precision(
            threshold_neg, precision_neg, target
        )

    metrics["rocauc"] = compute_rocauc(y_fact, y_pred)
    metrics["prauc(average_precision)"] = compute_prauc(y_fact, y_pred)

    npos = int(y_fact.sum())
    nneg = len(y_fact) - npos
    metrics["npos"] = npos
    metrics["nneg/npos"] = nneg / npos
    
    # all_thh = list(np.arange(0, 1, 0.01))
    # f1list_by_all_thhs = [skm.f1_score(df[y_fact], (df[y_pred] > thh)) for thh in all_thh]
    # maxindex = np.argmax(f1list_by_all_thhs)
    # metrics["f1_max"] = f1list_by_all_thhs[maxindex]
    # metrics["f1_max_thh"] = all_thh[maxindex]
    
    return metrics
