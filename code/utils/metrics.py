import numpy as np
from typing import List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

# --------------- GENERAL METRICS --------------------------------
def accuracy(model_answers, correct_answers):
    if len(model_answers) != len(correct_answers):
        raise ValueError("Model answers and correct answers must have the same length.")
    
    correct_count = sum(1 for ma, ca in zip(model_answers, correct_answers) if ma == ca)
    return correct_count / len(correct_answers)

def random_accuracy(questions):
    if not questions:
        return 0.0  # Avoid division by zero
    return sum(1 / options for options in questions) / len(questions)


def min_max_norm(scores: List[float]):
        """
        Performs min-max normalization of scores.
        Parameters:
            scores (List[float]): List of scores to normalize.
        Returns:
            List[float]: Normalized scores.
        """
        scores_arr = np.asarray(scores)
        mask = ~np.isnan(scores_arr)

        scaled = np.empty_like(scores_arr, dtype=float)
        scaled[:] = np.nan  # fill output with NaN

        scaled[mask] = MinMaxScaler().fit_transform(scores_arr[mask].reshape(-1, 1)).flatten()
        return scaled

def skip_target_nans(target, estimator):
    newt, newe = [], []
    for t, e in zip(target, estimator):
        if np.isnan(t):
            continue
        newt.append(t)
        newe.append(e)
    return newt, newe

class Stats:
    """
    Computes basic statistics (mean, median, std) for a list of scores.
    """
    def __init__(self, name):
        self.normalize_min_max = False
        self.digits = 6
        self.name = name
        functions = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "min": np.min,
            "max": np.max,
            "percentile_99": lambda x: np.percentile(x, 99)
        }
        self.function = functions[name]

    def __str__(self):
        return self.name

    def __call__(self, estimator: List[float], target: List[float]=None) -> float:
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")
        
        estimator = np.asarray(estimator)
        estimator = estimator[~np.isnan(estimator)]

        return round(self.function(estimator), self.digits)

# --------------- UE Metrics ----------------------
class ECE:
    """
    Expected Calibration Error (ECE) metric. Only applicable to binary quality metrics.
    """
    def __init__(self, normalize_min_max=False, n_bins=10):
        self.normalize_min_max = normalize_min_max
        self.n_bins = n_bins
        self.digits = 3

    def __str__(self):
        return "ece"

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Performs min-max normalization of scores.
        Parameters:
            scores (List[float]): List of scores to normalize.
        Returns:
            List[float]: Normalized scores.
        """

        scores = np.asarray(scores).reshape(-1, 1)

        return np.array(MinMaxScaler().fit_transform(scores).flatten())

    def __call__(self, estimator: List[float], target: List[float]) -> float:
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")
        
        estimator = np.asarray(estimator)
        target = np.asarray(target)

        # if self.normalize_min_max:
        #     confidences = -estimator
        #     confidences = self.normalize_scores(confidences)
        # else:
        # assert not np.any(estimator < -0.1) and not np.any(estimator > 1.1), f"ECE: min: {estimator.min()}, max: {estimator.max()}"
        confidences = 1-estimator
        # assert not np.any(confidences < -0.1) and not np.any(confidences > 1.1), "ECE: Confidences must be in [0, 1]"

        bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        ece, N = 0.0, len(confidences)
        for i in range(self.n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = (
                (confidences > lo) & (confidences <= hi)
                if i > 0
                else (confidences >= lo) & (confidences <= hi)
            )
            if not np.any(in_bin):
                continue
            acc_bin = np.mean(target[in_bin])
            conf_bin = np.mean(confidences[in_bin])
            ece += (np.sum(in_bin) / N) * abs(acc_bin - conf_bin)

        return round(ece, self.digits)
    
class Brier:
    """
    Computes the Brier score, which is a measure of the accuracy of probabilistic predictions.
    Adapted from https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/ue_metrics/roc_auc.py
    
    Args:
        estimator (List[float]): List of predicted probabilities.
        target (List[int]): List of true binary labels (0 or 1).
        
    Returns:
        float: Brier score.
    """
    def __init__(self, normalize_min_max=False):
        super().__init__()
        self.normalize_min_max = normalize_min_max
        self.digits = 3

    def __str__(self):
        return "brier"

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Performs min-max normalization of scores.
        Parameters:
            scores (List[float]): List of scores to normalize.
        Returns:
            List[float]: Normalized scores.
        """

        scores = np.asarray(scores).reshape(-1, 1)

        return np.array(MinMaxScaler().fit_transform(scores).flatten())
        
    def __call__(self, estimator: List[float], target: List[int]):
        if len(estimator) != len(target):
            raise ValueError("Estimator and target must have the same length.")

        estimator = np.asarray(estimator)
        target = np.asarray(target)

        # if self.normalize_min_max:
        #     confidences = -estimator
        #     confidences = self.normalize_scores(confidences)
        # else:
        confidences = 1-estimator
        assert not np.any(confidences < -0.1) and not np.any(confidences > 1.1), "Brier: Confidences must be in [0, 1]"
        
        confidences = np.clip(confidences, 0.0, 1.0)
        return round(brier_score_loss(target, confidences), self.digits)


class ROCAUC:
    """
    Computes the ROC AUC score, which is a measure of the accuracy of probabilistic predictions.
    Adapted from https://github.com/IINemo/lm-polygraph/blob/main/src/lm_polygraph/ue_metrics/roc_auc.py
    
    Args:
        estimator (List[float]): List of predicted probabilities.
        target (List[int]): List of true binary labels (0 or 1).
        
    Returns:
        float: ROC AUC score.
    """
    is_ood_metric = True

    def __str__(self):
        return "auroc"

    def preprocess_inf(self, x, array):
        if not np.isinf(x):
            return x
        elif x > 0:
            return array.max() + 1
        else:
            return array.min() - 1

    def __call__(self, estimator: List[float], target: List[int]):
        assert len(estimator) == len(target), "Estimator and target must have the same length."
        estimator = [self.preprocess_inf(x, estimator) for x in estimator]
        target = [not t for t in target]
        t, e = skip_target_nans(target, estimator)
        return roc_auc_score(t, e)