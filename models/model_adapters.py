from models.dixon_coles import DixonColes
from models.xgboost import XGBoostModel
from features.feature_builder import FULL_XGBOOST_FEATURES, convert_string_col_to_category, convert_int_col_to_category
from typing import Any
import polars as pl
import numpy as np

def dc_predict_fn(dc_model: DixonColes, row: dict[str, Any]) -> np.ndarray:
    """Adapts DixonColes.predict to the uniform predict_fn signature
    expected by walk_forward.

    Parameters
    ----------
    dc_model : DixonColes
        Fitted Dixon-Coles model instance for the current training window.
    row : dict[str, Any]
        A single match record, as produced by DataFrame.iter_rows(named=True).

    Returns
    -------
    np.ndarray
        Predicted probabilities in [home, draw, away] order.
    """
    return dc_model.predict(row["home_team"], row["away_team"])


def dc_model_info(dc_model: DixonColes, home: str, away: str, row: dict[str, Any]) -> dict[str, Any]:
    """Extracts Dixon-Coles team strength and shared model parameters for
    the current matchup, for inclusion in walk-forward evaluation records.

    Parameters
    ----------
    dc_model : DixonColes
        Fitted Dixon-Coles model instance for the current training window.
    home : str
        Home team name.
    away : str
        Away team name.
    row : dict[str, Any]
        A single match record, as produced by DataFrame.iter_rows(named=True).

    Returns
    -------
    dict[str, Any]
        Attack and defense ratings, plus the gamma and rho
        parameters.
    """
    info: dict[str, Any] = {}
    if hasattr(dc_model, "team_ratings"):
        info.update({
            "home_attack": dc_model.team_ratings[home]["attack"],
            "away_attack": dc_model.team_ratings[away]["attack"],
            "home_defense": dc_model.team_ratings[home]["defense"],
            "away_defense": dc_model.team_ratings[away]["defense"],
        })
    if hasattr(dc_model, "gamma"):
        info["gamma"] = dc_model.gamma
    if hasattr(dc_model, "rho"):
        info["rho"] = dc_model.rho
    return info


def xgboost_predict_fn(xgboost_model: XGBoostModel, 
                       row: dict[str, Any]) -> np.ndarray:
    """Adapts XGBoostModel.predict to the uniform predict_fn signature
    expected by walk_forward.

    Parameters
    ----------
    xgboost_model : XGBoostModel
        Fitted XGBoostModel instance for the current training window.
    row : dict[str, Any]
        A single match record, as produced by DataFrame.iter_rows(named=True).

    Returns
    -------
    np.ndarray
        Predicted probabilities in [home, draw, away] order.
    """
    df = pl.DataFrame([row])
    df = convert_string_col_to_category(df, "league")
    df = convert_int_col_to_category(df, "season")
    return xgboost_model.predict(df)


def xgboost_model_info(xgboost_model: XGBoostModel, home: str, away: str, row: dict[str, Any]) -> None:
    return


def dc_factory() -> DixonColes:
    """Zero-argument factory that builds a fresh, unfitted Dixon-Coles
    model instance for walk_forward to fit on each training window.

    Returns
    -------
    DixonColes
        New DixonColes instance.
    """
    return DixonColes(fit_options={"disp": False, "maxiter": 1000})


def xgboost_factory() -> XGBoostModel:
    """Zero-argument factory that builds a fresh, unfitted XGBoost
    model instance for walk_forward to fit on each training window.

    Returns
    -------
    XGBoostModel
        New XGBoostModel instance.
    """
    return XGBoostModel(FULL_XGBOOST_FEATURES)