import numpy as np
import polars as pl
import logging
logger = logging.getLogger(__name__)

class Elo:
    def __init__(self, k:float = 20.0, home_field_advantage:float = 60.0) -> None:
        """Elo system for rating teams based on match outcomes.

        Parameters
        ----------
        k : float, optional
            K-factor for Elo rating updates, by default 20.0
        home_field_advantage : float, optional
            Home field advantage in Elo rating, by default 60.0
        """
        self.k = k
        self.home_field_advantage = home_field_advantage
        self.team_ratings = {}
        self._fitted = False
        logger.info("Initialized %s(k=%s, home_field_advantage=%s)", type(self).__name__, self.k, self.home_field_advantage)

        
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculates the expected score for a match between two teams.

        Parameters
        ----------
        rating_a : float
            Rating of the first team.
        rating_b : float
            Rating of the second team.

        Returns
        -------
        float
            Expected score for the first team.
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


    def update_rating(self, rating_a: float, rating_b: float, score_a: float) -> float:
        """Updates the rating of the first team based on the match outcome.

        Parameters
        ----------
        rating_a : float
            Rating of the first team.
        rating_b : float
            Rating of the second team.
        score_a : float
            Actual score for the first team.

        Returns
        -------
        float
            Updated rating for the first team.
        """
        expected_a = self.expected_score(rating_a, rating_b)
        new_rating_a = rating_a + self.k * (score_a - expected_a)
        return new_rating_a

    
    def reset_ratings(self) -> None:
        """Resets the ratings for all teams.
        """
        self.team_ratings = {}
        self._fitted = False


    def update_dataframe(self, data:pl.DataFrame, team_col:str = "home_team", opponent_col:str = "away_team") -> pl.DataFrame:
        """Updates the DataFrame with home and away Elo ratings for each match.

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
            DataFrame with updated Elo ratings.
        """
        elo_data = {"home_elo": [], "away_elo": []}
        home_elo = []
        away_elo = []
        if self._fitted:
            logger.warning("Elo ratings have already been fitted. Resetting ratings before updating DataFrame.")
            self.reset_ratings()

        for row in data.iter_rows(named=True):
            home_team = row[team_col]
            away_team = row[opponent_col]

            if home_team not in self.team_ratings:
                self.team_ratings[home_team] = 1500.0
            if away_team not in self.team_ratings:
                self.team_ratings[away_team] = 1500.0

            home_elo.append(self.team_ratings[home_team])
            away_elo.append(self.team_ratings[away_team])

            home_rating = self.team_ratings[home_team] + self.home_field_advantage
            away_rating = self.team_ratings[away_team]

            if row["home_goals"] > row["away_goals"]: 
                score_home, score_away = 1, 0
            elif row["home_goals"] == row["away_goals"]: 
                score_home, score_away = 0.5, 0.5
            else:  
                score_home, score_away = 0, 1

            new_home_rating = self.update_rating(home_rating, away_rating, score_home)
            new_away_rating = self.update_rating(away_rating, home_rating, score_away)

            self.team_ratings[home_team] = new_home_rating - self.home_field_advantage
            self.team_ratings[away_team] = new_away_rating

        self._fitted = True
        return data.with_columns(
            pl.Series("home_elo", home_elo),
            pl.Series("away_elo", away_elo))