import math
from scipy.stats import norm
import numpy as np
from itertools import combinations
from scipy.stats import ttest_ind


# compare accuracies of two models on the same dataset
def mcnemar_from_counts(n01, n10, correction=True):
    """
    Standard McNemar test (large sample, chi square with 1 df).
    n01: count where A incorrect, B correct
    n10: count where A correct, B incorrect
    """
    discordant = n01 + n10
    if discordant == 0:
        return float("nan"), float("nan")

    diff = abs(n01 - n10)
    if correction:
        diff = max(diff - 1, 0)

    chi2 = (diff ** 2) / discordant if discordant > 0 else float("nan")
    if math.isnan(chi2):
        return chi2, float("nan")

    z = math.sqrt(chi2)
    p = 2.0 * (1.0 - norm.cdf(z))
    return chi2, p

def single_model_bootstrap_metric(
    metric_fn,
    estimates,
    targets,
    n_bootstrap=2000,
    ci_level=0.95,
    random_state=42,
):
    """
    Bootstrap confidence interval for a single model's metric.

    Parameters
    ----------
    metric_fn : callable
        metric_fn(estimates, targets) -> float
    estimates, targets : array-like
        Model predictions and targets
    n_bootstrap : int
        Number of bootstrap resamples
    ci_level : float
        Confidence level (e.g., 0.95)
    random_state : int or None

    Returns
    -------
    results : dict with keys:
        - observed_metric
        - ci (lower, upper)
        - bootstrap_metrics
    """
    rng = np.random.default_rng(random_state)
    estimates = np.asarray(estimates)
    targets = np.asarray(targets)
    n = len(targets)

    # Observed metric
    observed_metric = metric_fn(estimates, targets)

    # Bootstrap resampling
    bootstrap_metrics = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_metrics[b] = metric_fn(estimates[idx], targets[idx])

    # Confidence interval
    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))

    return {
        "observed_metric": observed_metric,
        "ci": (lower, upper),
        "std_bootstrap_metrics": np.std(bootstrap_metrics),
    }

# pairwise comparison of auroc, ece, brier
def paired_bootstrap_metric(
    metric_fn,
    estimates_list,
    targets,
    n_bootstrap=2000,
    ci_level=0.95,
    random_state=42,
):
    """
    Paired bootstrap comparison of metrics.

    Parameters
    ----------
    metric_fn : callable
        Function with signature metric_fn(estimates, targets) -> float
    estimates_list : list of array-like
        List of prediction arrays, all of same length
    targets : array-like
        Ground truth labels, length n
    n_bootstrap : int
        Number of bootstrap resamples
    ci_level : float
        Confidence level (e.g. 0.95)
    random_state : int or None

    Returns
    -------
    results : dict
        Keys are (i, j) estimator index pairs.
        Values are dicts with:
            - observed_diff
            - p_value
            - ci (lower, upper)
            - bootstrap_diffs
    """

    rng = np.random.default_rng(random_state)

    targets = np.asarray(targets)
    estimates_list = [np.asarray(e) for e in estimates_list]

    n = len(targets)
    assert all(len(e) == n for e in estimates_list)

    # observed metrics
    observed = [metric_fn(e, targets) for e in estimates_list]

    results = {}

    for i, j in combinations(range(len(estimates_list)), 2):
        diffs = np.empty(n_bootstrap)

        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            m_i = metric_fn(estimates_list[i][idx], targets[idx])
            m_j = metric_fn(estimates_list[j][idx], targets[idx])
            diffs[b] = m_i - m_j

        # confidence interval
        alpha = 1 - ci_level
        lower = np.percentile(diffs, 100 * alpha / 2)
        upper = np.percentile(diffs, 100 * (1 - alpha / 2))

        # two-sided p-value
        p_value = 2 * min(
            np.mean(diffs <= 0),
            np.mean(diffs >= 0)
        )

        results[(i, j)] = {
            "observed_diff": observed[i] - observed[j],
            "p_value": p_value,
            "ci": (lower, upper),
            "bootstrap_diffs": diffs,
        }

    return results

def unpaired_bootstrap_metric(
    metric_fn,
    estimates_a,
    targets_a,
    estimates_b,
    targets_b,
    n_bootstrap=2000,
    ci_level=0.95,
    random_state=42,
):
    """
    Unpaired (independent) bootstrap comparison of metrics.

    Parameters
    ----------
    metric_fn : callable
        metric_fn(estimates, targets) -> float
    estimates_a, targets_a : array-like
        Model A predictions and targets
    estimates_b, targets_b : array-like
        Model B predictions and targets
    n_bootstrap : int
        Number of bootstrap resamples
    ci_level : float
        Confidence level (e.g. 0.95)
    random_state : int or None

    Returns
    -------
    results : dict with keys:
        - observed_diff
        - p_value
        - ci (lower, upper)
        - bootstrap_diffs
    """

    rng = np.random.default_rng(random_state)

    estimates_a = np.asarray(estimates_a)
    targets_a = np.asarray(targets_a)
    estimates_b = np.asarray(estimates_b)
    targets_b = np.asarray(targets_b)

    n_a = len(targets_a)
    n_b = len(targets_b)

    assert len(estimates_a) == n_a
    assert len(estimates_b) == n_b

    # observed metrics
    m_a_obs = metric_fn(estimates_a, targets_a)
    m_b_obs = metric_fn(estimates_b, targets_b)
    observed_diff = m_a_obs - m_b_obs

    diffs = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)

        m_a = metric_fn(estimates_a[idx_a], targets_a[idx_a])
        m_b = metric_fn(estimates_b[idx_b], targets_b[idx_b])

        diffs[b] = m_a - m_b

    # confidence interval
    alpha = 1 - ci_level
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    # two-sided p-value
    p_value = 2 * min(
        np.mean(diffs <= 0),
        np.mean(diffs >= 0)
    )

    return {
        "observed_diff": observed_diff,
        "observed_a": m_a_obs,
        "observed_b": m_b_obs,
        "p_value": p_value,
        "ci": (lower, upper),
        "bootstrap_diffs": diffs,
    }

def compare_estimator_outputs(
    estimator_outputs_modified,
    estimator_outputs_default,
    alpha=0.05,
):
    # Convert to numpy arrays
    x = np.asarray([val for val in estimator_outputs_modified if not np.isnan(val)], dtype=float)
    assert np.all(~np.isnan(x)), "estimator_outputs_modified contains NaN values"
    y = np.asarray([val for val in estimator_outputs_default if not np.isnan(val)], dtype=float)
    assert np.all(~np.isnan(y)), "estimator_outputs_default contains NaN values"
    print(f"X,Y length after NaN removal: {len(x)}, {len(y)}")

    if x.size < 2 or y.size < 2:
        raise ValueError("Both inputs must contain at least two values.")

    # Means
    avg_x = np.mean(x)
    avg_y = np.mean(y)

    # Relative delta average
    if avg_y == 0:
        relative_delta_avg = np.nan
    else:
        relative_delta_avg = (avg_x - avg_y) / avg_y

    # Welch's t-test (unequal variances)
    t_stat, p_value = ttest_ind(x, y, equal_var=False)

    return {
        "avg_outputs_modified": avg_x,
        "avg_outputs_default": avg_y,
        "relative_delta_avg": relative_delta_avg,
        "t_statistic": t_stat,
        "p_value": p_value,
        "significant": p_value < alpha,
    }


