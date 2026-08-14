from typing import Literal
import numpy as np
from scipy.stats import norm
from numpy.typing import NDArray
from collections.abc import Sequence

BlockSize = Literal["sqrt", "square_root", "n^(1/2)", "n**(1/2)", "default",
                     "cubed", "cubic", "cubic_root", "cube_root", "n^(1/3)", "n**(1/3)"]

def block_bootstrap(data: Sequence[float] | NDArray[np.float64], block_size: int | BlockSize | None = None, n_bootstraps: int = 1000) -> NDArray[np.float64]:
    """Moving block bootstrap for the mean of a time series.

    Resamples `data` by concatenating randomly-selected overlapping
    blocks (with replacement) to preserve local autocorrelation
    structure, then computes the mean of each bootstrap sample.

    Parameters
    ----------
    data : Sequence[float] | NDArray[np.float64]
        The time series to bootstrap.
    block_size : int, BlockSize, or None
        Length of each block. If None, defaults to 10. Also accepts
        "sqrt"/"n^(1/2)"-style aliases for n**0.5, and "cubed"/"n^(1/3)"-
        style aliases for n**(1/3) (see source for the exact accepted
        strings).
    n_bootstraps : int
        Number of bootstrap resamples to draw.

    Returns
    -------
    NDArray[np.float64]
        Array of shape (n_bootstraps,) containing the mean of each
        bootstrap sample.
    """
    n = len(data)
    if block_size is None:
        block_size = 10
    elif block_size in ("sqrt", "square_root", "n^(1/2)", "n**(1/2)", "default"):
        block_size = int(n**0.5)
    elif block_size in ("cubed", "cubic", "cubic_root", "cube_root", "n^(1/3)", "n**(1/3)"):
        block_size = int(n**(1/3))
    elif isinstance(block_size, int):
        pass
    else:
        raise ValueError(f"Unrecognized block_size: {block_size!r}")
    n_blocks = n // block_size
    indices = np.arange(n - block_size + 1)
    bootstrapped_mean_samples = []

    for bootstrap in range(n_bootstraps):
        # Randomly select starting indices of blocks with replacement
        selected_blocks_indices = np.random.choice(indices, size=n_blocks, replace=True)
        # Concatenate the selected blocks to form a bootstrap sample
        bootstrap_sample = np.concatenate([data[i:i + block_size] for i in selected_blocks_indices])
        bootstrapped_mean_samples.append(np.mean(bootstrap_sample))

    return np.array(bootstrapped_mean_samples)


def autocovariance(x: Sequence[float] | NDArray[np.float64], lag: int) -> float:
    """Calculate the autocovariance of a time series at a given lag.
    
    Parameters
    ----------
    x : Sequence[float] | NDArray[np.float64]
        The time series.
    lag : int
        The lag at which to calculate the autocovariance.

    Returns
    -------
    float
        The autocovariance at the specified lag.
    """
    # Autocovariance at lag k is defined as:
    # \sum_{i=k+1}^{n} (x_i - \bar{x})(x_{i-k} - \bar{x}) / n
    n = len(x)
    x_bar = np.mean(x)
    x = np.asarray(x)
    return np.sum((x[:-lag] - x_bar) * (x[lag:] - x_bar)) / n


def diebold_mariano_test(loss_diff: Sequence[float] | NDArray[np.float64], h: int = 1, alpha: float = 0.05, verbose: bool = False) -> tuple[float, float, bool]:
    """Diebold-Mariano test for equal predictive accuracy of two forecasts.

    Tests the null hypothesis that two forecasts have the same expected
    loss, given their loss differential series.

    Parameters
    ----------
    loss_diff : Sequence[float] | NDArray[np.float64]
        Loss differential series (loss_a - loss_b) between two
        competing forecasts, one value per matched observation.
    h : int
        Forecast horizon. Determines how many autocovariance lags
        (1 to h-1) are included in the long-run variance estimate.
    alpha : float
        Significance level for the two-sided test.
    verbose : bool
        If True, print diagnostic output including per-lag
        autocovariance contributions.

    Returns
    -------
    DM_estimate : float
        The Diebold-Mariano test statistic.
    p_value : float
        Two-sided p-value under the standard normal approximation.
    significant : bool
        Whether the null hypothesis is rejected at level `alpha`.
    """
    n = len(loss_diff)
    mean_loss_diff = np.mean(loss_diff)
    gamma0 = np.var(loss_diff)
    autocovariances_sum = 0.0
    lag_contributions = []
    for lag in range(1, h):
        gamma_k = autocovariance(loss_diff, lag)
        weighted = 2 * (1 - lag / h) * gamma_k
        lag_contributions.append((lag, gamma_k, weighted))
        autocovariances_sum += gamma_k

    DM_estimate = mean_loss_diff / np.sqrt(gamma0 / n + 2 * autocovariances_sum / n)
    z_two_sided = norm.ppf(1 - alpha / 2)
    p_value = 2 * (1 - norm.cdf(abs(DM_estimate)))
    significant = abs(DM_estimate) > z_two_sided
    var_d_bar = gamma0 / n + 2 * autocovariances_sum / n

    if verbose:
        print(f"n (matches): {n}")
        print(f"mean loss diff (a - b): {mean_loss_diff:.6f}")
        print(f"gamma0 (lag-0 var): {gamma0:.6f}")
        print(f"sum weighted autocovariance: {autocovariances_sum:.6f}")
        print(f"var(d_bar): {var_d_bar:.6f}")
        print(f"DM statistic: {DM_estimate:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"critical z ({alpha=}): {z_two_sided:.4f}")
        print(f"significant?: {significant}")
        print("per-lag contributions (lag, gamma_k, weighted):")
        for lag, gamma_k, weighted in lag_contributions:
            print(f"lag {lag:2d}: gamma_k={gamma_k:+.6f}  weighted={weighted:+.6f}")

    return DM_estimate, p_value, significant