import polars as pl 
from exceptions import InvalidMatchDataError

BASE_FEATURE_COLUMNS = ["league", "season",
"home_form_npxGD", "away_form_npxGD",
"home_form_corner_xGD", "away_form_corner_xGD",
"home_form_adjusted_GD", "away_form_adjusted_GD",
"home_avg_npxGD", "away_avg_npxGD",
"home_avg_corner_xGD", "away_avg_corner_xGD",
"home_avg_adjusted_GD", "away_avg_adjusted_GD",
"home_form_ppda_zone_adj", "away_form_ppda_zone_adj",
"home_form_field_tilt", "away_form_field_tilt",
"home_std_field_tilt_for", "away_std_field_tilt_for",
"home_std_ppda_zone_adj_for", "away_std_ppda_zone_adj_for",
"home_std_ppda_zone_adj_against", "away_std_ppda_zone_adj_against",
"home_std_xGOT_for", "away_std_xGOT_for",
"home_std_xGOT_against", "away_std_xGOT_against",
"home_form_touches_box_for", "away_form_touches_box_for",
"home_form_touches_box_against", "away_form_touches_box_against",
"home_std_touches_opp_box_for", "away_std_touches_opp_box_for",
"home_std_touches_opp_box_against", "away_std_touches_opp_box_against"]

DC_PARAMS = ["home_attack", "home_defense", 
             "away_attack", "away_defense", 
             "home_DC_xg", "away_DC_xg",
             "gamma", "rho", 
             "dc_prob_home", "dc_prob_draw", "dc_prob_away"]

FULL_XGBOOST_FEATURES = BASE_FEATURE_COLUMNS + DC_PARAMS

DC_WALK_FORWARD_COLS_TO_RECORD = [
    "home_goals", "away_goals",

    "home_form_npxgf",
    "home_form_npxga",
    "home_form_xgot_for",
    "home_form_xgot_against",
    "home_form_touches_box_for",
    "home_form_touches_box_against",
    "home_form_ppda_zone_adj",
    "home_form_field_tilt",
    "home_form_corner_xg_for",
    "home_form_corner_xg_against",
    "home_form_adjusted_goals_for",
    "home_form_adjusted_goals_against",

    "away_form_npxgf",
    "away_form_npxga",
    "away_form_xgot_for",
    "away_form_xgot_against",
    "away_form_touches_box_for",
    "away_form_touches_box_against",
    "away_form_ppda_zone_adj",
    "away_form_field_tilt",
    "away_form_corner_xg_for",
    "away_form_corner_xg_against",
    "away_form_adjusted_goals_for",
    "away_form_adjusted_goals_against",

    "home_std_npxG_for",
    "home_std_xGOT_for",
    "home_std_touches_opp_box_for",
    "home_std_ppda_zone_adj_for",
    "home_std_field_tilt_for",
    "home_std_corner_xG_for",
    "home_std_adjusted_goals_for",
    "home_std_npxG_against",
    "home_std_xGOT_against",
    "home_std_touches_opp_box_against",
    "home_std_ppda_zone_adj_against",
    "home_std_field_tilt_against",
    "home_std_corner_xG_against",
    "home_std_adjusted_goals_against",

    "away_std_npxG_for",
    "away_std_xGOT_for",
    "away_std_touches_opp_box_for",
    "away_std_ppda_zone_adj_for",
    "away_std_field_tilt_for",
    "away_std_corner_xG_for",
    "away_std_adjusted_goals_for",
    "away_std_npxG_against",
    "away_std_xGOT_against",
    "away_std_touches_opp_box_against",
    "away_std_ppda_zone_adj_against",
    "away_std_field_tilt_against",
    "away_std_corner_xG_against",
    "away_std_adjusted_goals_against",

    "league",
    "season"]

def convert_string_col_to_category(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Casts a string column to a Polars Enum based on its sorted unique values.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing the column to convert.
    col : str
        Name of the column to cast into a categorical (Enum) type.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with the specified column cast to pl.Enum.
    """
    categories = df[col].unique().sort().to_list()
    df = df.with_columns(pl.col(col).cast(pl.Enum(categories)))
    return df


def convert_int_col_to_category(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Casts an int column to a Polars Enum based on its sorted unique values.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing the column to convert.
    col : str
        Name of the column to cast into a categorical (Enum) type.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with the specified column cast to pl.Enum.
    """
    categories = [str(s) for s in sorted(df[col].unique().to_list())]
    df = df.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Enum(categories)))
    return df


def create_differences(df: pl.DataFrame,
                       stat_for: str, stat_against: str,
                       new_col_name: str) -> pl.DataFrame:
    """Creates a new column in the DataFrame that is the difference between two specified columns.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing the columns to produce a difference from.
    stat_for : str
        Statistic to subtract from.
    stat_against : str
        Statistic used to subtract.
    new_col_name : str
        Name of the new column to store the computed difference.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with an additional difference column. 
    """
    df = df.with_columns((pl.col(stat_for) - pl.col(stat_against)).alias(new_col_name))
    return df


def reconstruct_and_add_DC_expected_goals(df: pl.DataFrame) -> pl.DataFrame:
    """Reconstructs Dixon-Coles expected goals for home and away teams from model parameters.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing the Dixon Coles params as columns.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with "home_DC_xg" and "away_DC_xg" columns added.
    """
    df = df.with_columns((pl.col("home_attack") + pl.col("away_defense") + pl.col("gamma"))
                         .exp().alias("home_DC_xg"))
    df = df.with_columns((pl.col("away_attack") + pl.col("home_defense"))
                             .exp().alias("away_DC_xg"))
    return df


def unpack_DC_prob_array(df: pl.DataFrame, col_name: str = "probs") -> pl.DataFrame:
    """Unpacks a list-valued Dixon-Coles probability column into separate outcome columns.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing a column of 3-element probability lists ordered as
        (home, draw, away).
    col_name : str, optional
        Name of the column holding the probability lists, by default "probs".

    Returns
    -------
    pl.DataFrame
        The input DataFrame with "col_name" replaced by three unnested columns:
        "dc_prob_home", "dc_prob_draw", and "dc_prob_away".
    """
    df = df.with_columns(pl.col(col_name).list.to_struct(fields=["dc_prob_home", "dc_prob_draw", "dc_prob_away"])).unnest("probs")
    return df


def create_xgboost_feature_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Builds the full set of XGBoost feature columns from raw match statistics.

    Parameters
    ----------
    df : pl.DataFrame
        Match data containing the relevant params.

    Returns
    -------
    pl.DataFrame
        The input DataFrame enriched with the full set of XGBoost feature
        columns defined in FULL_XGBOOST_FEATURES.
    """
    df = create_differences(df, stat_for="home_form_npxgf", stat_against="home_form_npxga", new_col_name="home_form_npxGD")
    df = create_differences(df, stat_for="away_form_npxgf", stat_against="away_form_npxga", new_col_name="away_form_npxGD")
    df = create_differences(df, stat_for="home_form_corner_xg_for", stat_against="home_form_corner_xg_against", new_col_name="home_form_corner_xGD")
    df = create_differences(df, stat_for="away_form_corner_xg_for", stat_against="away_form_corner_xg_against", new_col_name="away_form_corner_xGD")
    df = create_differences(df, stat_for="home_form_adjusted_goals_for", stat_against="home_form_adjusted_goals_against", new_col_name="home_form_adjusted_GD")
    df = create_differences(df, stat_for="away_form_adjusted_goals_for", stat_against="away_form_adjusted_goals_against", new_col_name="away_form_adjusted_GD")

    df = create_differences(df, stat_for="home_std_npxG_for", stat_against="home_std_npxG_against", new_col_name="home_avg_npxGD")
    df = create_differences(df, stat_for="away_std_npxG_for", stat_against="away_std_npxG_against", new_col_name="away_avg_npxGD")
    df = create_differences(df, stat_for="home_std_corner_xG_for", stat_against="home_std_corner_xG_against", new_col_name="home_avg_corner_xGD")
    df = create_differences(df, stat_for="away_std_corner_xG_for", stat_against="away_std_corner_xG_against", new_col_name="away_avg_corner_xGD")
    df = create_differences(df, stat_for="home_std_adjusted_goals_for", stat_against="home_std_adjusted_goals_against", new_col_name="home_avg_adjusted_GD")
    df = create_differences(df, stat_for="away_std_adjusted_goals_for", stat_against="away_std_adjusted_goals_against", new_col_name="away_avg_adjusted_GD")


    df = convert_string_col_to_category(df, "league")
    df = convert_int_col_to_category(df, "season")
    
    df = reconstruct_and_add_DC_expected_goals(df)
    df = unpack_DC_prob_array(df)

    validate_columns(df)
    return df


def validate_columns(df: pl.DataFrame) -> None:
    """Checks if required features are present in dataframe.

    Parameters
    ----------
    df : pl.DataFrame
        Data containing features.

    Raises
    ------
    InvalidMatchDataError
        If a feature isn't found, InvalidMatchDataError is raised.
    """
    missing = set(FULL_XGBOOST_FEATURES) - set(df.columns)
    if missing:
        raise InvalidMatchDataError(f"Missing columns: {sorted(missing)}")