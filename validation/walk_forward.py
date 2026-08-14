from collections.abc import Callable
from typing import Any
import numpy as np
import polars as pl
from enum import IntEnum

class Outcome(IntEnum):
    HOME_WIN = 0
    DRAW = 1
    AWAY_WIN = 2


def default_outcome_idx(row: dict[str, Any], 
                        home_goals_col: str, 
                        away_goals_col: str) -> Outcome:
    """Determines the outcome of a match based on goal counts.

    Parameters
    ----------
    row : dict[str, Any]
        A single match record, as produced by DataFrame.iter_rows(named=True)
    home_goals_col : str
        Name of the column corresponding to home-team goals.
    away_goals_col : str
        Name of the column corresponding to away-team goals.

    Returns
    -------
    Outcome
        The outcome of the match.
    """
    if row[home_goals_col] > row[away_goals_col]:
        return Outcome.HOME_WIN
    elif row[away_goals_col] > row[home_goals_col]:
        return Outcome.AWAY_WIN
    else:
        return Outcome.DRAW


def _fit_model(model_factory: Callable[[], Any],
               train_data: pl.DataFrame, 
               date_col: str, 
               fit_kwargs: dict[str, Any] | None = None) -> Any:
    """Fits a model on the given training data.

    Parameters
    ----------
    model_factory : Callable[[], Any]
        Zero-argument callable that returns a fresh, unfitted model
        instance. Called once per training window. Must return a new
        instance on every call, not a shared/reused one; reusing an
        instance would carry fitted state over from the previous window
        into the next fit, silently breaking the isolation walk-forward
        evaluation depends on.
    train_data : pl.DataFrame
        The data on which to fit the model.
    date_col : str
        Name of the column containing the match dates.
    fit_kwargs : dict[str, Any] | None = None
        Extra keyword arguments forwarded as-is to the model's fit method.

    Returns
    -------
    Any
        The fitted model.
    """
    model = model_factory()
    fit_kwargs = fit_kwargs or {}
    try:
        model.fit(train_data, **fit_kwargs)
        return model
    except Exception as e:
        print(f"Walk forward fit failed on window ending "
              f"{train_data[date_col][-1]} ({train_data.height} rows): {e}")
        return None
 
 
def _known_teams(model: Any) -> set[str] | None:
    """Returns the set of known team names from the model.

    Parameters
    ----------
    model : Any
        The model instance.

    Returns
    -------
    set[str] | None
        Set of known team names.
    """
    if hasattr(model, "result") and isinstance(model.result, dict):
        return set(model.result.keys()).difference({"gamma", "rho"})
    return None
 
 
def _score_window(model: Any, test_data: pl.DataFrame, predict_fn: Callable,
                  outcome_fn: Callable[[dict[str, Any]], Outcome],
                  home_col: str, away_col: str, date_col: str, 
                  records: list[dict[str, Any]]) -> None:
    """Scores a single evaluation window.

    Parameters
    ----------
    model : Any
        The model instance.
    test_data : pl.DataFrame
        Subset of full match data for the current evaluation window.
    predict_fn : Callable
        Adapts a model's native prediction call into the uniform
        (model, row) signature this module expects, pulling whatever
        fields it needs from row itself (e.g. team names) to call the
        model however its own predict method is actually shaped. Must
        return probabilities in [home, draw, away] order and
        is required for rps, brier, log_loss to score correctly.
    outcome_fn : Callable[[dict[str, Any]], Outcome]
        The function to compute the outcome of a match.
    home_col : str
        Name of the column containing the home team names.
    away_col : str
        Name of the column containing the away team names.
    date_col : str
        Name of the column containing the match dates.
    records : list[dict[str, Any]]
        List to which the evaluation results will be appended.
    """
    teams_ok = _known_teams(model)
    unknown_teams = 0
    for row in test_data.iter_rows(named=True):
        home, away = row[home_col], row[away_col]
        if teams_ok is not None and (home not in teams_ok or away not in teams_ok):
            unknown_teams += 1
        try:
            probs = np.asarray(predict_fn(model, row), dtype=float)
        except Exception as e:
            print(f"Walk forward predict failed for {home} vs {away} "
                  f"on {row[date_col]}: {e}")
            raise
        records.append({"date": row[date_col],
                        "home_team": home,
                        "away_team": away,
                        "probs": probs.tolist(),
                        "outcome": outcome_fn(row)})
    if unknown_teams:
        print(f"Walk forward: {unknown_teams} rows in this window had unknown team(s)")
        print(f"Empirical priors used to fill in missing team strengths.")


def _unpack_cols(col_names: dict[str, str] | None) -> tuple[str, str, str, str, str]:
    """Unpacks the column names for the five required columns.

    Parameters
    ----------
    col_names : dict[str, str] | None
        Maps the five required column names, "date_col", "home_col",
        "away_col", "home_goals_col", "away_goals_col", to the actual
        column names in data. If None, defaults to
        {"date_col": "match_date", "home_col": "home_team",
        "away_col": "away_team", "home_goals_col": "home_goals",
        "away_goals_col": "away_goals"}. If provided, all five keys are
        required; a partial mapping raises a KeyError rather than
        falling back to defaults for the missing keys.

    Returns
    -------
    tuple[str, str, str, str, str]
        Tuple of the five column names. 
    """
    if col_names is None:
        col_names = {"date_col": "match_date",
                     "home_col": "home_team",
                     "away_col": "away_team",
                     "home_goals_col": "home_goals",
                     "away_goals_col": "away_goals"}
    return (col_names["date_col"], col_names["home_col"], col_names["away_col"], col_names["home_goals_col"], col_names["away_goals_col"])


def walk_forward_by_date(
        data: pl.DataFrame,
        model_factory: Callable[[], Any],
        predict_fn: Callable,
        col_names: dict[str, str] | None = None,
        min_train_matches: int = 380,
        rolling_window: int | None = None,
        verbose: bool = True,
        fit_kwargs: dict[str, Any] | None = None) -> pl.DataFrame:
    """Performs walk-forward validation on the given data. Each 
    walk-forward evaluation step size is determined by the matchday dates and 
    matches played in the matchday.

    Parameters
    ----------
    data : pl.DataFrame
        Full match history. Must contain the five columns referenced by
        col_names (or the defaults, if col_names is None). Any
        additional columns are passed through untouched to the model's
        fit method, so columns needed only by a specific model's
        fitting logic don't need to be known.
        Does not need to be pre-sorted by date as each function 
        sorts by date_col internally.
    model_factory : Callable[[], Any]
        Zero-argument callable that returns a fresh, unfitted model
        instance. Called once per training window. Must return a new
        instance on every call, not a shared/reused one; reusing an
        instance would carry fitted state over from the previous window
        into the next fit, silently breaking the isolation walk-forward
        evaluation depends on.
    predict_fn : Callable
        Adapts a model's native prediction call into the uniform
        (model, row) signature this module expects, pulling whatever
        fields it needs from row itself (e.g. team names) to call the
        model however its own predict method is actually shaped. Must
        return probabilities in [home, draw, away] order and
        is required for rps, brier, log_loss to score correctly.
    col_names : dict[str, str] | None = None
        Maps the five required column names, "date_col", "home_col",
        "away_col", "home_goals_col", "away_goals_col", to the actual
        column names in data. If None, defaults to
        {"date_col": "match_date", "home_col": "home_team",
        "away_col": "away_team", "home_goals_col": "home_goals",
        "away_goals_col": "away_goals"}. If provided, all five keys are
        required; a partial mapping raises a KeyError rather than
        falling back to defaults for the missing keys.
    min_train_matches : int = 380
        Number of matches used for model burn in; that is, before any
        evaluation begins and primarily for model parameters to converge 
        initially. Subsequent refits grow (or slide, if
        rolling_window is set) from this starting point.
    rolling_window : int | None = None
        Number of most-recent matches to train on at each refit. If
        None, training uses an expanding window — every past match is
        kept. If set, training uses a fixed-size sliding window — matches
        older than rolling_window are dropped as new ones arrive. Note
        this is redundant (and can discard information a smoother
        approach would keep) for models that already apply their own
        time-decay weighting internally
    verbose : bool = True
        Print one progress line per refit: the cutoff/last
        date trained through, cumulative matches trained on, 
        and cumulative records scored so far.
    fit_kwargs : dict[str, Any] | None = None
        Extra keyword arguments forwarded as-is to the model's fit method.

    Returns
    -------
    pl.DataFrame
        Dataframe containing the evaluation results 
        (match date, home/away teams, predicted/actual outcomes).
    """
    date_col, home_col, away_col, home_goals_col, away_goals_col = _unpack_cols(col_names)
    data = data.sort(date_col)
    outcome_fn = (lambda row: default_outcome_idx(row, home_goals_col, away_goals_col))
    n = data.height
    records = []
    cutoffs = data[date_col].unique().sort().to_list()
    windows = [(d, data.filter(pl.col(date_col) == d)) for d in cutoffs]
    train_end = 0
    for cutoff, test_data in windows:
        if train_end < min_train_matches:
            train_end += test_data.height
            continue
        train_start = 0 if rolling_window is None else max(0, train_end - rolling_window)
        train_data = data.slice(train_start, train_end - train_start)
        model = _fit_model(model_factory, train_data, date_col, fit_kwargs)
        if model is not None:
            _score_window(model, test_data, predict_fn, outcome_fn, home_col,
                            away_col, date_col, records)
        train_end += test_data.height
        if verbose:
            print(f"[walk_forward] refit through {cutoff} -- "
                    f"{train_end}/{n} matches trained on, {len(records)} scored so far")
    return pl.DataFrame(records)


def walk_forward_by_refit_window(
        data: pl.DataFrame, 
        model_factory: Callable[[], Any],
        predict_fn: Callable,
        col_names: dict[str, str] | None = None,
        min_train_matches: int = 380,
        refit_every: int = 10,
        rolling_window: int | None = None,
        verbose: bool = True,
        fit_kwargs: dict[str, Any] | None = None) -> pl.DataFrame:
    """Performs walk-forward validation by refitting the model at fixed intervals.
    Unlike the per matchday walk forward validation, this approach 
    can fit matches of the same matchday in different walk forward steps if the 
    refit window is small enough. 

    Parameters
    ----------
    data : pl.DataFrame
        Full match history. Must contain the five columns referenced by
        col_names (or the defaults, if col_names is None). Any
        additional columns are passed through untouched to the model's
        fit method, so columns needed only by a specific model's
        fitting logic don't need to be known.
        Does not need to be pre-sorted by date as each function 
        sorts by date_col internally.
    model_factory : Callable[[], Any]
        Zero-argument callable that returns a fresh, unfitted model
        instance. Called once per training window. Must return a new
        instance on every call, not a shared/reused one; reusing an
        instance would carry fitted state over from the previous window
        into the next fit, silently breaking the isolation walk-forward
        evaluation depends on.
    predict_fn : Callable
        Adapts a model's native prediction call into the uniform
        (model, row) signature this module expects, pulling whatever
        fields it needs from row itself (e.g. team names) to call the
        model however its own predict method is actually shaped. Must
        return probabilities in [home, draw, away] order and
        is required for rps, brier, log_loss to score correctly.
    col_names : dict[str, str] | None = None
        Maps the five required column names, "date_col", "home_col",
        "away_col", "home_goals_col", "away_goals_col", to the actual
        column names in data. If None, defaults to
        {"date_col": "match_date", "home_col": "home_team",
        "away_col": "away_team", "home_goals_col": "home_goals",
        "away_goals_col": "away_goals"}. If provided, all five keys are
        required; a partial mapping raises a KeyError rather than
        falling back to defaults for the missing keys.
    min_train_matches : int = 380
        Number of matches used for model burn in; that is, before any
        evaluation begins and primarily for model parameters to converge 
        initially. Subsequent refits grow (or slide, if
        rolling_window is set) from this starting point.
    refit_every : int = 10
        Number of matches between refits. Also determines the size of
        each test window, since exactly this many matches are scored
        under a given fit before the next refit occurs. Note that the 
        final window may contain fewer matches.
    rolling_window : int | None = None
        Number of most-recent matches to train on at each refit. If
        None, training uses an expanding window — every past match is
        kept. If set, training uses a fixed-size sliding window — matches
        older than rolling_window are dropped as new ones arrive. Note
        this is redundant (and can discard information a smoother
        approach would keep) for models that already apply their own
        time-decay weighting internally
    verbose : bool = True
        Print one progress line per refit: the cutoff/last
        date trained through, cumulative matches trained on, 
        and cumulative records scored so far.
    fit_kwargs : dict[str, Any] | None = None
        Extra keyword arguments forwarded as-is to the model's fit method.

    Returns
    -------
    pl.DataFrame
        Dataframe containing the evaluation results 
        (match date, home/away teams, predicted/actual outcomes).
    """
    date_col, home_col, away_col, home_goals_col, away_goals_col = _unpack_cols(col_names)
    data = data.sort(date_col)
    outcome_fn = (lambda row: default_outcome_idx(row, home_goals_col, away_goals_col))
    n = data.height
    records = []
    train_end = min_train_matches
    while train_end < n:
        train_start = 0 if rolling_window is None else max(0, train_end - rolling_window)
        train_data = data.slice(train_start, train_end - train_start)
        test_data = data.slice(train_end, refit_every)
        if test_data.height == 0:
            break
        model = _fit_model(model_factory, train_data, date_col, fit_kwargs)
        if model is not None:
            _score_window(model, test_data, predict_fn, outcome_fn, home_col,
                            away_col, date_col, records)
        train_end += refit_every
        if verbose:
            last_date = test_data[date_col][-1]
            print(f"[walk_forward] refit through {last_date}  {min(train_end, n)}/{n} matches trained on, {len(records)} scored so far")

    return pl.DataFrame(records)


def walk_forward(
        data: pl.DataFrame, 
        model_factory: Callable[[], Any],
        predict_fn: Callable,
        col_names: dict[str, str] | None = None,
        min_train_matches: int = 380,
        refit_every: int = 10,
        rolling_window: int | None = None,
        group_refits_by_date: bool = False,
        verbose: bool = True,
        fit_kwargs: dict[str, Any] | None = None) -> pl.DataFrame:
    """Performs walk-forward validation on the given data. The walk-forward
    step can be determined by the match date or a fixed interval.
    
    Parameters
    ----------
    data : pl.DataFrame
        Full match history. Must contain the five columns referenced by
        col_names (or the defaults, if col_names is None). Any
        additional columns are passed through untouched to the model's
        fit method, so columns needed only by a specific model's
        fitting logic don't need to be known.
        Does not need to be pre-sorted by date as each function 
        sorts by date_col internally.
    model_factory : Callable[[], Any]
        Zero-argument callable that returns a fresh, unfitted model
        instance. Called once per training window. Must return a new
        instance on every call, not a shared/reused one; reusing an
        instance would carry fitted state over from the previous window
        into the next fit, silently breaking the isolation walk-forward
        evaluation depends on.
    predict_fn : Callable
        Adapts a model's native prediction call into the uniform
        (model, row) signature this module expects, pulling whatever
        fields it needs from row itself (e.g. team names) to call the
        model however its own predict method is actually shaped. Must
        return probabilities in [home, draw, away] order and
        is required for rps, brier, log_loss to score correctly.
    col_names : dict[str, str] | None = None
        Maps the five required column names, "date_col", "home_col",
        "away_col", "home_goals_col", "away_goals_col", to the actual
        column names in data. If None, defaults to
        {"date_col": "match_date", "home_col": "home_team",
        "away_col": "away_team", "home_goals_col": "home_goals",
        "away_goals_col": "away_goals"}. If provided, all five keys are
        required; a partial mapping raises a KeyError rather than
        falling back to defaults for the missing keys.
    min_train_matches : int = 380
        Number of matches used for model burn in; that is, before any
        evaluation begins and primarily for model parameters to converge 
        initially. Subsequent refits grow (or slide, if
        rolling_window is set) from this starting point.
    refit_every : int = 10
        Number of matches between refits. Also determines the size of
        each test window, since exactly this many matches are scored
        under a given fit before the next refit occurs. Note that the
        final window may contain fewer matches.
    rolling_window : int | None = None
        Number of most-recent matches to train on at each refit. If
        None, training uses an expanding window — every past match is
        kept. If set, training uses a fixed-size sliding window — matches
        older than rolling_window are dropped as new ones arrive. Note
        this is redundant (and can discard information a smoother
        approach would keep) for models that already apply their own
        time-decay weighting internally
    group_refits_by_date : bool = False
        Refit at every matchday boundary, so all matches sharing a date
        are always scored under the same fitted model. 
        If False, refit every refit_every matches regardless of date,
        which can split a single matchday's fixtures across
        two different fits. 
    verbose : bool = True
        Print one progress line per refit: the cutoff/last
        date trained through, cumulative matches trained on, 
        and cumulative records scored so far.
    fit_kwargs : dict[str, Any] | None = None
        Extra keyword arguments forwarded as-is to the model's fit method.

    Returns
    -------
    pl.DataFrame
        Dataframe containing the evaluation results 
        (match date, home/away teams, predicted/actual outcomes).
    """
    if group_refits_by_date:
        records = walk_forward_by_date(data=data, model_factory=model_factory, predict_fn=predict_fn, col_names=col_names, min_train_matches=min_train_matches, rolling_window=rolling_window, verbose=verbose, fit_kwargs=fit_kwargs)
    else:
        records = walk_forward_by_refit_window(data=data, model_factory=model_factory, predict_fn=predict_fn, col_names=col_names, min_train_matches=min_train_matches, refit_every=refit_every, rolling_window=rolling_window, verbose=verbose, fit_kwargs=fit_kwargs)

    return records