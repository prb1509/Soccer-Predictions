import numpy as np
from numpy.typing import NDArray
from collections.abc import Sequence
import logging
logger = logging.getLogger(__name__)

def rps(probs: Sequence[float] | NDArray[np.float64], outcome_idx: int) -> float:
    """Ranked Probability Score, ordinal order: [home, draw, away].
    0 = perfect, 1 = worst.

    Parameters
    ----------
    probs : Sequence[float] | NDArray[np.float64]
        Array of predicted probabilities for each outcome.
    outcome_idx : int
        Index of the actual outcome.

    Returns
    -------
    float
        The ranked probability score.
    """
    # Definition of RPS:
    # \sum_{i=1}^{r-1}\sum_{j=1}^{i}(p_j - o_j)^2 / (r-1)
    probs = np.asarray(probs)
    logger.debug("Sum of probabilities: %f", sum(probs))
    cum_probs = np.cumsum(probs)
    actual = np.zeros(len(probs))
    actual[outcome_idx] = 1
    cum_actual = np.cumsum(actual)
    logger.debug("RPS: probs=%s, cum_probs=%s, actual=%s, cum_actual=%s", probs, cum_probs, actual, cum_actual)
    return float(np.sum((cum_probs - cum_actual)**2) / (len(probs) - 1))


def brier(probs: Sequence[float] | NDArray[np.float64], outcome_idx: int) -> float:
    """Multi-class Brier score. 0 = perfect, 2 = worst.
    
    Parameters
    ----------
    probs : Sequence[float] | NDArray[np.float64]
        Array of predicted probabilities for each outcome.
    outcome_idx : int
        Index of the actual outcome.

    Returns
    -------
    float
        The Brier score.
    """
    # Definition of Brier score:
    # \sum_{i=1}^{r}(p_i - o_i)^2
    probs = np.asarray(probs)
    logger.debug("Sum of probabilities: %f", sum(probs))
    actual = np.zeros(len(probs))
    actual[outcome_idx] = 1
    logger.debug("Brier score: probs=%s, actual=%s", probs, actual)
    return float(np.sum((probs - actual)**2))


def log_loss(probs: Sequence[float] | NDArray[np.float64], outcome_idx: int, eps: float = 1e-15) -> float:
    """Log loss (cross-entropy) for the single actual outcome.
    0 = perfect, unbounded above without the clipping.
    
    Parameters
    ----------
    probs : Sequence[float] | NDArray[np.float64]
        Array of predicted probabilities for each outcome.
    outcome_idx : int
        Index of the actual outcome.
    eps : float, optional
        Small value to clip probabilities, by default 1e-15. Prevents taking the logarithm of zero.
    Returns
    -------
    float
        The log loss (ignorance) score.
    """
    # Definition of log loss:
    # -log(p_i), where p_i is the predicted probability assigned to
    # the actual outcome i 
    probs = np.asarray(probs)
    logger.debug("Sum of probabilities: %f", sum(probs))
    p = np.clip(probs[outcome_idx], eps, 1 - eps)
    logger.debug("Log loss: probs=%s, outcome_idx=%d, p=%f", probs, outcome_idx, p)
    return float(-np.log(p))