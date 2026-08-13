import numpy as np
import polars as pl
from enum import IntEnum

class Outcome(IntEnum):
    HOME_WIN = 0
    DRAW = 1
    AWAY_WIN = 2


def default_outcome_idx(row, home_goals_col, away_goals_col):
    if row[home_goals_col] > row[away_goals_col]:
        return Outcome.HOME_WIN
    elif row[away_goals_col] > row[home_goals_col]:
        return Outcome.AWAY_WIN
    else:
        return Outcome.DRAW


def _fit_model(model_factory, train_data, date_col, fit_kwargs=None):
    model = model_factory()
    fit_kwargs = fit_kwargs or {}
    try:
        model.fit(train_data, **fit_kwargs)
        return model
    except Exception as e:
        print(f"Walk forward fit failed on window ending "
              f"{train_data[date_col][-1]} ({train_data.height} rows): {e}")
        raise
 
 
def _known_teams(model):
    if hasattr(model, "result") and isinstance(model.result, dict):
        return set(model.result.keys()).difference_update({"gamma", "rho"})
    return None
 
 
def _score_window(model, test_data, predict_fn, outcome_fn, home_col, away_col,
                   date_col, records):
    teams_ok = _known_teams(model)
    dropped_unknown_teams = 0
    for row in test_data.iter_rows(named=True):
        home, away = row[home_col], row[away_col]
        if teams_ok is not None and (home not in teams_ok or away not in teams_ok):
            dropped_unknown_teams += 1
            continue
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
    if dropped_unknown_teams:
        print(f"Walk forward: dropped {dropped_unknown_teams} rows in this window due to unknown team(s)")


def _unpack_cols(col_names):
    if col_names is None:
        col_names = {"date_col": "match_date",
                     "home_col": "home_team",
                     "away_col": "away_team",
                     "home_goals_col": "home_goals",
                     "away_goals_col": "away_goals"}
    return (col_names["date_col"], col_names["home_col"], col_names["away_col"], col_names["home_goals_col"], col_names["away_goals_col"])


def walk_forward_by_date(data: pl.DataFrame, model_factory, predict_fn,
                 col_names = None,
                 min_train_matches: int = 380,
                 rolling_window: int | None = None,
                 verbose=True,
                 fit_kwargs = None):
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


def walk_forward_by_refit_window(data: pl.DataFrame, model_factory, predict_fn,
                 col_names = None,
                 min_train_matches: int = 380,
                 refit_every: int = 10,
                 rolling_window: int | None = None,
                 verbose=True,
                 fit_kwargs = None):
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


def walk_forward(data: pl.DataFrame, model_factory, predict_fn,
                 col_names = None,
                 min_train_matches: int = 380,
                 refit_every: int = 10,
                 rolling_window: int | None = None,
                 group_refits_by_date: bool = False,
                 verbose=True,
                 fit_kwargs = None):
 
    if group_refits_by_date:
        records = walk_forward_by_date(data=data, model_factory=model_factory, predict_fn=predict_fn, col_names=col_names, min_train_matches=min_train_matches, rolling_window=rolling_window, verbose=verbose, fit_kwargs=fit_kwargs)
    else:
        records = walk_forward_by_refit_window(data=data, model_factory=model_factory, predict_fn=predict_fn, col_names=col_names, min_train_matches=min_train_matches, refit_every=refit_every, rolling_window=rolling_window, verbose=verbose, fit_kwargs=fit_kwargs)

    return records