import numpy as np
import polars as pl
import logging
logger = logging.getLogger(__name__)

class Pi:
    def __init__(self, gamma:float = 0.7, 
    lambda_:float = 0.035, c:float = 3) -> None:
        """Pi system for rating teams based on match outcomes.

        Reference:
        Constantinou, A. C., & Fenton, N. E. (2013). Determining the level of
        ability of football teams by dynamic ratings based on the relative
        discrepancies in scores between adversaries. Journal of Quantitative
        Analysis in Sports, 9(1), 37-50. https://doi.org/10.1515/jqas-2012-0036

        Default hyperparameters (gamma=0.7, lambda_=0.035, c=3) are taken from
        the original paper's optimized values on EPL data.

        Parameters
        ----------
        gamma : float, optional
            Learning rate which determines to what extent the newly
            acquired information based on home performance influences a team's
            away rating and vice versa, by default 0.7
        lambda_ : float, optional
            Learning rate which determines the impact of the goal difference on the rating update, by default 0.035
        c : float, optional
            Scale parameter for the goal difference calculation and error, by default 3
        """
        self.home_ratings = {}
        self.away_ratings = {}
        self.gamma = gamma
        self.lambda_ = lambda_
        self.c = c
        self._fitted = False
        logger.info("Initialized %s(gamma=%s, lambda_=%s, c=%s)", type(self).__name__, self.gamma, self.lambda_, self.c)


    def goal_difference_home(self, home_rating: float) -> float:
        """Calculates the goal difference for the home team against an average opponent based on its rating.

        Parameters
        ----------
        home_rating : float
            Home team's rating.

        Returns
        -------
        float
            Expected goal difference for the home team.
        """
        if home_rating < 0:
            gd = -(10**np.abs((home_rating)/self.c) - 1)
        else:
            gd = 10**(home_rating/self.c) - 1
        return gd


    def goal_difference_away(self, away_rating: float) -> float:
        """Calculates the goal difference for the away team against an average opponent based on its rating.

        Parameters
        ----------
        away_rating : float
            Away team's rating.

        Returns
        -------
        float
            Expected goal difference for the away team.
        """
        if away_rating < 0:
            gd = -(10**np.abs((away_rating) / self.c) - 1)
        else:
            gd = 10**(away_rating / self.c) - 1
        return gd


    def expected_goal_difference(self, home_rating: float, away_rating: float) -> float:
        """Calculates the expected goal difference between two teams based on their ratings.

        Parameters
        ----------
        home_rating : float
            Home team's rating.
        away_rating : float
            Away team's rating.

        Returns
        -------
        float
            Expected goal difference between the two teams.
        """
        home_gd = self.goal_difference_home(home_rating)
        away_gd = self.goal_difference_away(away_rating)
        return home_gd - away_gd


    def update_rating(self, home_ratings: list[float], away_ratings: list[float], actual_gd: float, expected_gd: float) -> tuple[list[float], list[float]]:
        """Updates the ratings of the home and away teams based on the actual and expected goal differences.

        Parameters
        ----------
        home_ratings : list[float]
            List containing the home team's home and away ratings in that order.
        away_ratings : list[float]
            List containing the away team's home and away ratings in that order.
        actual_gd : float
            Actual goal difference.
        expected_gd : float
            Expected goal difference.

        Returns
        -------
        tuple[list[float], list[float]]
            Updated ratings for the home and away teams.
        """
        home_ratings[0] += self.lambda_ * (actual_gd - expected_gd) 
        home_ratings[1] += self.gamma * (actual_gd - expected_gd) * self.lambda_ 
        away_ratings[1] -= self.lambda_ * (actual_gd - expected_gd)
        away_ratings[0] -= self.gamma * (actual_gd - expected_gd) * self.lambda_ 
        return home_ratings, away_ratings

    def reset_ratings(self) -> None:
        """Resets the ratings for all teams.
        """
        self.home_ratings = {}
        self.away_ratings = {}
        self._fitted = False

    
    def update_dataframe(self, data:pl.DataFrame, team_col:str = "home_team", opponent_col:str = "away_team") -> pl.DataFrame:
        """Updates the DataFrame with home and away pi ratings for each match.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame containing match data with home and away teams.
        team_col : str, optional
            Column name for the home team, by default "home_team"
        opponent_col : str, optional
            Column name for the away team, by default "away_team"

        Returns
        -------
        pl.DataFrame
            DataFrame with updated Pi ratings.
        """
        home_pi = []
        away_pi = []
        if self._fitted:
            logger.warning("Pi ratings have already been fitted. Resetting ratings before updating DataFrame.")
            self.reset_ratings()

        for row in data.iter_rows(named=True):
            home_team = row[team_col]
            away_team = row[opponent_col]

            if home_team not in self.home_ratings:
                self.home_ratings[home_team] = 0.0

            if away_team not in self.away_ratings:
                self.away_ratings[away_team] = 0.0

            if home_team not in self.away_ratings:
                self.away_ratings[home_team] = 0.0
            
            if away_team not in self.home_ratings:
                self.home_ratings[away_team] = 0.0

            home_rating = self.home_ratings[home_team]
            away_rating = self.away_ratings[away_team]
            
            home_pi.append(home_rating)
            away_pi.append(away_rating)

            expected_gd = self.expected_goal_difference(home_rating, away_rating)

            actual_gd = row["home_goals"] - row["away_goals"]

            updated_home_rating, updated_away_rating = self.update_rating([home_rating, self.away_ratings[home_team]], 
                                                        [self.home_ratings[away_team], away_rating], 
                                                        actual_gd, expected_gd)

            self.home_ratings[home_team] = updated_home_rating[0]
            self.away_ratings[home_team] = updated_home_rating[1]
            self.home_ratings[away_team] = updated_away_rating[0]
            self.away_ratings[away_team] = updated_away_rating[1]

        self._fitted = True
        return data.with_columns(
            pl.Series("home_pi", home_pi),
            pl.Series("away_pi", away_pi))