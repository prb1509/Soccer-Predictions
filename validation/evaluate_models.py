from models.model_adapters import xgboost_predict_fn, dc_predict_fn, dc_model_info, dc_factory, xgboost_factory
from validation.walk_forward import walk_forward
from validation.metrics import rps, log_loss, brier
from features.feature_builder import DC_WALK_FORWARD_COLS_TO_RECORD
import polars as pl
from collections.abc import Callable

def add_metric_to_result(result: pl.DataFrame, metric: Callable,
                          col_name: str) -> pl.DataFrame:
    """Applies a metric function row-wise to a walk-forward result and adds it as a new column.

    Parameters
    ----------
    result : pl.DataFrame
        Walk forward result containing "probs" and "outcome" columns.
    metric : Callable
        Metric function taking (probs, outcome) and returning a score.
    col_name : str
        Name of the new column to store the computed metric values under.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional column containing the per-row metric values.
    """
    result = result.with_columns(pl.struct(["probs", "outcome"])
                  .map_elements(lambda s: metric(s["probs"], s["outcome"]), return_dtype=pl.Float64)
                  .alias(col_name))
    return result


def compute_model_metrics(walk_forward_result: pl.DataFrame) -> pl.DataFrame:
    """Computes RPS, Brier score, and log loss for each prediction in a walk forward result.

    Parameters
    ----------
    walk_forward_result : pl.DataFrame
        Walk forward generated predictions.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with "rps", "brier", and "log_loss" columns added.
    """
    walk_forward_result = add_metric_to_result(walk_forward_result, rps, "rps")
    walk_forward_result = add_metric_to_result(walk_forward_result, brier, "brier")
    walk_forward_result = add_metric_to_result(walk_forward_result, log_loss, "log_loss")
    return walk_forward_result


def print_model_performance(walk_forward_result: pl.DataFrame) -> None:
    """Prints model RPS, Brier score, and log loss after walk forward evaluation.

    Parameters
    ----------
    walk_forward_result : pl.DataFrame
        Walk forward generated predictions.
    """
    print(f"Model walk forward rps: {walk_forward_result['rps'].mean()}")
    print(f"Model walk forward brier score: {walk_forward_result['brier'].mean()}")
    print(f"Model walk forward log loss: {walk_forward_result['log_loss'].mean()}")


def run_walk_forward_DC(dataset: pl.DataFrame, xi: float, 
                        min_train_matches: int = 380,
                        refit_every: int = 10,
                        rolling_window: int | None = None, 
                        group_refits_by_date: bool = False) -> pl.DataFrame:
    """Runs walk-forward validation using the Dixon-Coles model.

    Parameters
    ----------
    dataset : pl.DataFrame
        Dataset of matches to walk forward over, used for both training and prediction.
    xi : float
        Time-decay rate controlling how quickly older matches lose influence.
    min_train_matches : int, optional
        Number of matches used for model burn in, by default 380.
    refit_every : int, optional
        Number of matches between refits. Also determines the size of
        each test window, by default 10
    rolling_window : int | None, optional
        Number of most-recent matches to train on at each refit. If
        None, training uses an expanding window: every past match is
        kept. By default None
    group_refits_by_date : bool, optional
        Refit at every matchday boundary, so all matches sharing a date
        are always scored under the same fitted model, by default False.
    
    Returns
    -------
    pl.DataFrame
        Walk forward results containing predictions, outcomes, and any extra
        recorded columns for each match.
    """
    wf_results = walk_forward(dataset,
            model_factory=dc_factory,
            predict_fn=dc_predict_fn,
            min_train_matches=min_train_matches,
            refit_every=refit_every,
            rolling_window=rolling_window,
            group_refits_by_date=group_refits_by_date,
            fit_kwargs={"xi": xi}, 
            extra_cols=DC_WALK_FORWARD_COLS_TO_RECORD,
            model_info_fn=dc_model_info)

    return wf_results


def run_walk_forward_xgboost(dataset: pl.DataFrame,
                             min_train_matches: int = 1000, 
                             refit_every:int = 10,
                             rolling_window: int | None = None, 
                             group_refits_by_date: bool = False) -> pl.DataFrame:
    """Runs walk-forward validation using the XGBoost model.

    Parameters
    ----------
    dataset : pl.DataFrame
        Dataset of matches to walk forward over, used for both training and prediction.
    min_train_matches : int, optional
        Number of matches used for model burn in, by default 1000.
    refit_every : int, optional
        Number of matches between refits. Also determines the size of
        each test window, by default 10
    rolling_window : int | None, optional
        Number of most-recent matches to train on at each refit. If
        None, training uses an expanding window: every past match is
        kept. By default None
    group_refits_by_date : bool, optional
        Refit at every matchday boundary, so all matches sharing a date
        are always scored under the same fitted model, by default False.
    
    Returns
    -------
    pl.DataFrame
        Walk forward results containing predictions, outcomes, and any extra
        recorded columns for each match.
    """
    wf_results = walk_forward(dataset,
        model_factory=xgboost_factory,
        predict_fn=xgboost_predict_fn,
        extra_cols=[],
        min_train_matches=min_train_matches,
        refit_every=refit_every,
        rolling_window=rolling_window,
        group_refits_by_date=group_refits_by_date, model_info_fn=None,)

    return wf_results