import numpy as np

def rps(probs, outcome_idx):
    """Ranked Probability Score, ordinal order: [home, draw, away].
    0 = perfect, 1 = worst."""
    # Definition of RPS:
    # \sum_{i=1}^{r-1}\sum_{j=1}^{i}(p_j - o_j)^2 / (r-1)
    cum_probs = np.cumsum(probs)
    actual = np.zeros(len(probs))
    actual[outcome_idx] = 1
    cum_actual = np.cumsum(actual)
    return float(np.sum((cum_probs - cum_actual) ** 2) / (len(probs) - 1))


def brier(probs, outcome_idx):
    """Multi-class Brier score. 0 = perfect, 2 = worst."""
    # Definition of Brier score:
    # \sum_{i=1}^{r}(p_i - o_i)^2
    probs = np.asarray(probs)
    actual = np.zeros(len(probs))
    actual[outcome_idx] = 1
    return float(np.sum((probs - actual) ** 2))


def log_loss(probs, outcome_idx, eps=1e-15):
    """Log loss (cross-entropy) for the single actual outcome.
    0 = perfect, unbounded above without the clipping."""
    # Definition of log loss:
    # -log(p_i), where p_i is the predicted probability assigned to
    # the actual outcome i 
    probs = np.asarray(probs)
    p = np.clip(probs[outcome_idx], eps, 1 - eps)
    return float(-np.log(p))