class SoccerPredictionsError(Exception):
    """Base class for all errors raised deliberately by this package.

    Catching this (instead of bare Exception) lets calling the code
    distinguish package specific errors from
    unrelated bugs or third-party (numpy/scipy/polars) errors.
    """


class InvalidMatchDataError(SoccerPredictionsError):
    """Match data is missing required columns, is empty, or contains
    values (goal counts, dates) that can't be used for fitting or
    scoring.

    Examples: fit() called with an empty DataFrame; a required
    column ("home_goals", "match_date", ...) missing; negative or
    NaN goal counts.
    """


class ModelNotFittedError(SoccerPredictionsError):
    """A method that requires a fitted model (predict, show_results,
    save_model) was called before fit().
    """


class InvalidParameterError(SoccerPredictionsError):
    """A constructor or function argument is out of its valid range.

    Examples: a non-negative rho_bound passed to DixonColes (bounds
    require rho_bound < 0); a non-positive refit_every or
    rolling_window passed to walk_forward; a block_size larger than
    the series length passed to block_bootstrap.
    """


class OptimizationError(SoccerPredictionsError):
    """The underlying scipy.optimize.minimize call failed to converge
    and the caller asked for strict failure rather than a logged
    warning.
    """