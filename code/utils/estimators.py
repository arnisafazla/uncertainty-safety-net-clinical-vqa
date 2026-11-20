import numpy as np
from collections import Counter
from typing import List, Tuple, Dict, Union
from abc import ABC, abstractmethod
from scipy.special import logsumexp

class BaseUE(ABC):
    def __init__(self, method_name: str = "PRO", normalize_options: bool = False) -> None:
        self.level = "sequence"
        self.normalize_options = normalize_options
        self.name = method_name
        self.requires_embeddings = False
        
    def __str__(self):
        return self.name

    @abstractmethod
    def __call__(self, stats: Dict[str, list]) -> np.ndarray:
        pass

# --------------- utilities ----------------------
def consistency(labels):
    if not labels:
        return 0.0
    # flatten labels
    labels = [label for sublist in labels for label in sublist]
    
    counts = Counter(labels)
    most_common_count = counts.most_common(1)[0][1]
    return 1 - most_common_count / len(labels)

def compute_probabilities_from_logprobs(logprobs: List[float]) -> float:
    NLL = np.mean(-np.array(logprobs))
    prob = np.exp(-NLL)
    return float(prob)

def shannon_entropy(labels: List[str]) -> float:
    if not labels:
        return 0.0
    # flatten labels
    labels = [label for sublist in labels for label in sublist]
    N = len(labels)
    counts = Counter(labels)  # counts per class
    probs = np.array([count / N for count in counts.values()])
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def shannon_entropy_of_distribution_no_clustering(
    probs: List[Tuple[str, List[Tuple[str, float]]]]
) -> float:
    # Compute sequence log probabilities
    logps = []

    for _, token_logprobs in probs:
        token_logprob_values = np.array([lp for _, lp in token_logprobs])

        # Sequence log probability
        logp = np.sum(token_logprob_values)

        # Skip pathological values (optional)
        if np.isfinite(logp):
            logps.append(logp)

    if len(logps) == 0:
        return 0.0

    logps = np.array(logps)

    # Normalize in log space
    logq = logps - logsumexp(logps)

    # Convert to probabilities
    q = np.exp(logq)

    # Compute entropy, ignoring zero-probability terms
    mask = q > 0
    entropy = -np.sum(q[mask] * logq[mask])

    return float(entropy)

def get_label_prob(logprobs) -> Union[float, None]:
    for i, (token, lp) in enumerate(logprobs):
        if "]" in token:
            if any(label in token for label in ["A", "B", "C", "D", "E"]):
                return lp  # invalid label
            elif i == 0:
                return np.nan  # no label before closing
            elif any(label in logprobs[i-1][0] for label in ["A", "B", "C", "D", "E"]):
                return logprobs[i-1][1]  # combine logprobs
    return np.nan

# ------------------------------------------ UE classes ----------------------------------------------
# For SE and SC
class DistributionUE(BaseUE):
    def __init__(self, method_name: str = "SE", normalize_options: bool = False) -> None:
        super().__init__(method_name, normalize_options)
        self.level = "sequence"
        self.stats_dependencies = "extracted_answers_per_sample"
        self.normalize_options = normalize_options
        self.name = method_name
        if self.name == "SE":
            self.method = shannon_entropy
        elif self.name == "SC":
            self.method = consistency

    def __call__(self, stats: Dict[str, list]) -> np.ndarray:
        """
        Estimates the Shannon entropy of the prediction given a list of predicted labels for the same input.

        Parameters:
            stats (Dict[str, list]): input statistics, which for multiple samples includes:
                * [[A, B, C, D], [A, B, B, B]] in 'extracted_labels_majority_voting'
                * [4, 4, 5, 3] in 'options_counts', which is how many options there are per each sample.
        Returns:
            np.ndarray: Shannon Entropy of each predicted class.
                Higher values indicate more uncertain samples.
        """
        labels_batch = stats["extracted_answers_per_sample"]
        assert isinstance(labels_batch, list)
        assert all(isinstance(labels, list) for labels in labels_batch)
        assert len(labels_batch[0][0]) <= 1, "DistributionUE only supports single-label classification."

        uncertainty_estimates = np.array([self.method(labels) for labels in labels_batch])

        if self.normalize_options:
            options_counts = stats["options_counts"]
            uncertainty_estimates = uncertainty_estimates / np.log(np.array(options_counts))
        return uncertainty_estimates

# For PRO
class DistributionOfProbUE(BaseUE):
    def __init__(self, method_name: str = "PRO", normalize_options: bool = False) -> None:
        super().__init__(method_name, normalize_options)
        self.level = "sequence"
        self.stats_dependencies = "raw_logprobs_per_sample"
        self.normalize_options = normalize_options
        self.name = method_name
        if self.name == "PRO":
            self.method = shannon_entropy_of_distribution_no_clustering

    def __call__(self, stats: Dict[str, list]) -> np.ndarray:
        """
        Estimates the Shannon entropy of the prediction given a list of predicted labels for the same input.

        Parameters:
            stats (Dict[str, list]): input statistics, which for multiple samples includes:
                * [[A, B, C, D], [A, B, B, B]] in 'extracted_labels_majority_voting'
                * [4, 4, 5, 3] in 'options_counts', which is how many options there are per each sample.
        Returns:
            np.ndarray: Shannon Entropy of each predicted class.
                Higher values indicate more uncertain samples.
        """
        raw_logprobs_per_sample = stats["raw_logprobs_per_sample"]
        assert isinstance(raw_logprobs_per_sample, list)
        
        uncertainty_estimates = np.array([self.method(logprobs) for logprobs in raw_logprobs_per_sample])

        if self.normalize_options:
            options_counts = stats["options_counts"]
            uncertainty_estimates = uncertainty_estimates / np.log(np.array(options_counts))
        return uncertainty_estimates

# For RDS and EE
class EmbeddingsBasedUE(BaseUE):
    def __init__(self, method_name: str = "RDS", normalize_options: bool = False, embedding_file: str = "embeddings-biomedical") -> None:
        super().__init__(method_name, normalize_options)
        self.level = "sequence"
        self.stats_dependencies = [f"{embedding_file}_per_sample"]
        self.normalize_options = normalize_options
        self.name = method_name if embedding_file == "embeddings" else f"{method_name}_{embedding_file}"
        self.requires_embeddings = True
        self.embedding_file = f"{embedding_file}_per_sample"

    def __call__(self, stats: Dict[str, list]) -> np.ndarray:
        """
        Estimates the Shannon entropy of the prediction given a list of predicted labels for the same input.

        Parameters:
            stats (Dict[str, list]): input statistics, which for multiple samples includes:
                * [[A, B, C, D], [A, B, B, B]] in 'extracted_labels_majority_voting'
                * [4, 4, 5, 3] in 'options_counts', which is how many options there are per each sample.
        Returns:
            np.ndarray: Shannon Entropy of each predicted class.
                Higher values indicate more uncertain samples.
        """
        embeddings = stats[self.embedding_file]
        assert isinstance(embeddings, list)

        if self.name == "RDSw":
            raw_logprobs_per_sample: List[List[Tuple[str, List[Tuple[str, float]]]]] = stats["raw_logprobs_per_sample"]
            assert isinstance(raw_logprobs_per_sample, list)
            probabilities_per_sample = []
            for sample_logprobs in raw_logprobs_per_sample:
                probabilities_per_shuffle = []
                for label, logprobs in sample_logprobs:
                    logprobs_list: List[float] = [token[1] for token in logprobs]
                    probabilities_per_shuffle.append(compute_probabilities_from_logprobs(logprobs_list))
                probabilities_per_sample.append(probabilities_per_shuffle)
        else:
            probabilities_per_sample = [[] for _ in embeddings]
        
        def radial_dispersion(embeds: list, probs: list = []) -> float:
            if len(embeds) == 0:
                return 0.0
            embeds = np.array(embeds)
            if probs:
                assert len(probs) == len(embeds)
                centroid = np.average(embeds, axis=0, weights=probs)
            else:
                centroid = np.mean(embeds, axis=0)
            distances = np.linalg.norm(embeds - centroid, ord=1, axis=1)
            if "EE" in self.name:
                return float(np.mean(np.sum((embeds - centroid)**2, axis=1)))

            if probs:
                return float(np.average(distances, weights=probs)) / 2
            else:
                return float(np.mean(distances)) / 2

        uncertainty_estimates = np.array([radial_dispersion(embed, probs=probs) for embed, probs in zip(embeddings, probabilities_per_sample)])
        if self.normalize_options:
            options_counts = stats["options_counts"]
            uncertainty_estimates = uncertainty_estimates / np.log(np.array(options_counts))
        return uncertainty_estimates

# For Label NLL
class LabelProbability(BaseUE):
    """
    Estimates the label-level uncertainty of a language model by calculating the
    probability of the generated label.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "LabelNLL"

    def __call__(self, stats: Dict[str, list]) -> np.ndarray:
        logprobs: List[List[List[Tuple[str, float]]]] = stats["raw_logprobs_per_sample"]
        label_log_likelihoods = [[get_label_prob(logprob[1]) for logprob in sample_logprobs] for sample_logprobs in logprobs]
        neg_label_log_likelihoods = [-np.nanmean(sample_nlls) for sample_nlls in label_log_likelihoods]
        # Filter out None entries
        return np.array(neg_label_log_likelihoods)

# For Max NLL and ANLL
class SequenceProbability(BaseUE):
    """
    Estimates the sequence-level uncertainty of a language model by calculating the
    log-probability of the generation with minus sign.
    It is calculated as the sum of log-probabilities in each token.
    Works only with whitebox models (initialized using lm_polygraph.utils.model.WhiteboxModel).
    """

    def __init__(self, aggregate: str, negative: bool = True, probabilities: bool = False) -> None:
        super().__init__()
        self.aggregate = aggregate
        self.negative = negative if not probabilities else False
        self.probabilities = probabilities

        class Option:
            def __init__(self, aggregate: str, negative: bool = True, probabilities: bool = False):
                self.aggregate = aggregate
                aggregate_functions = {
                    "mean": np.mean,
                    "max": np.max,
                    "min": np.min,
                    "sum": np.sum,
                }
                self.func = aggregate_functions.get(aggregate)
                if self.func is None:
                    raise ValueError(f"Unknown aggregation method: {aggregate}")

                self.negative = negative if not probabilities else False
                self.probabilities = probabilities

            def __str__(self) -> str:
                name = self.aggregate.replace("mean", "A").title()
                name += "N" if self.negative else "P"
                name += "LL" if not self.probabilities else "P"
                return name

            def __call__(self, log_likelihoods: np.ndarray) -> float:
                assert self.func is not None, "Aggregation function is not defined."

                log_likelihoods = np.array(log_likelihoods)
                if self.probabilities:
                    log_likelihoods = np.exp(log_likelihoods)
                else:
                    log_likelihoods = -log_likelihoods if self.negative else log_likelihoods

                return float(self.func(log_likelihoods))
                
        self.option = Option(aggregate, negative=negative, probabilities=probabilities)
        self.name = str(self.option)

    def __call__(self, stats: Dict[str, list]) -> np.ndarray:
        """
        Estimates the minus log-probability of each sample in input statistics.

        Parameters:
            stats (Dict[str, np.ndarray]): input statistics, which for multiple samples includes:
                * log p(y_i | y_<i, x) in 'greedy_log_likelihoods'
        Returns:
            np.ndarray: minus log probabilities for each sample.
                Higher values indicate more uncertain samples.
        """
        log_likelihoods = stats["logprobs_list_per_sample"]
        outputs = []
        for sample_log_likelihoods in log_likelihoods:
            outputs.append(np.mean([self.option(log_likelihood) for log_likelihood in sample_log_likelihoods if log_likelihood is not None and len(log_likelihood) > 0]).item())

        return np.array(outputs)
