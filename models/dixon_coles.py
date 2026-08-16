from typing import TypedDict
from scipy.optimize import minimize
import numpy as np
import polars as pl
from scipy.stats import poisson
from datetime import datetime, date
from exceptions import OptimizationError, InvalidParameterError, ModelNotFittedError
import logging
logger = logging.getLogger(__name__)

class DixonColes:
    def __init__(self, fit_options: dict | None = None, 
                 rho_init: float | None = None, 
                 rho_bound: float | None = None) -> None:
        """Initialize the model configuration.

        Parameters
        ----------
        fit_options : dict | None, optional
            Options forwarded to the optimizer: scipy.optimize.minimize; by default None
        rho_init : float | None, optional
            Initial value for the rho parameter, by default None
        rho_bound : float | None, optional
            Lower bound for the rho parameter, by default None
        """
        if fit_options is None:
            self.fit_options = {"disp": True, "maxiter": 1000}
        else:
            self.fit_options = fit_options

        if rho_init is None:
            self.rho_init = -0.05
        else:
            self.rho_init = rho_init

        if rho_bound is None:
            self.rho_bound = -0.5
        elif rho_bound >= 0:
            raise InvalidParameterError("rho_bound must be negative.")
        else:
            self.rho_bound = rho_bound


    def fit(self, data: pl.DataFrame, reference_date: datetime | date | None = None, 
            xi: float = 0) -> None:
        """Fit the Dixon-Coles model to historical match data.
        Estimates per-team attack and defense ratings along with the
        home advantage parameter (gamma) and the low-score
        correlation parameter (rho) by maximizing the (potentially
        time-weighted) Dixon-Coles log-likelihood via 
        constrained (sum of attack parameters = 0) optimization.

        Parameters
        ----------
        data : pl.DataFrame
            Match results containing at least the columns
            "home_team", "away_team", "home_goals", "away_goals",
            and "match_date".
        reference_date : datetime | date | None, optional
            Date from which time-decay weights are computed.
            The default, None, sets this to the latest match date in 
            the training data.
        xi : float, optional
            Time-decay rate controlling how quickly older matches
            lose influence. xi=0 disables decay which is enabled
            by default.
        """
        logger.info("Fitting Dixon-Coles model with xi=%f", xi)
        self._store_team_info(data)
        if reference_date is None:
            reference_date = max(data["match_date"])
        self._compute_time_decay_weights(reference_date, xi, data)
        raw_result = self._optimize_parameters(data)
        self.team_ratings, self.gamma, self.rho = self._clean_raw_result(raw_result)
        self.result = self.team_ratings | {"gamma": self.gamma, "rho": self.rho}
        logger.debug("Fitted model parameters: %s", self.result)


    def predict(self, home_team: str, away_team: str, 
                max_goals: int = 10) -> tuple[float, float, float]:
        """Predict outcome probabilities for a single fixture.
        Combines the fitted attack/defense ratings for both teams
        with the home advantage and rho parameters to compute
        expected goals for each side, then derives match outcome
        probabilities from the Dixon-Coles-adjusted Poisson score
        distribution. Teams that were not present in the training
        data are assumed to be newly promoted teams and thus fall 
        back to the average rating of the weakest rated
        teams.

        Parameters
        ----------
        home_team : str
            Name of the home team
        away_team : str
            Name of the away team
        max_goals : int, optional
            Maximum number of goals to consider in the probability 
            calculation, by default 10

        Returns
        -------
        tuple[float, float, float]
            Probabilities of a home win, draw, and away win,
            respectively; sum to 1.
        """
        if max_goals <= 0:
            logger.error("Invalid max_goals parameter: %d. Must be positive.", max_goals)
            raise InvalidParameterError("max_goals must be positive.")
        if not hasattr(self, "team_ratings"):
            logger.error("Model has not been fitted yet. Call fit() before predict(). Model is missing team_ratings.")
            raise ModelNotFittedError("Model must be fitted before making predictions.")
        if not hasattr(self, "gamma"):
            logger.error("Model has not been fitted yet. Call fit() before predict(). Model is missing gamma.")
            raise ModelNotFittedError("Model must be fitted before making predictions.")
        if not hasattr(self, "rho"):
            logger.error("Model has not been fitted yet. Call fit() before predict(). Model is missing rho.")
            raise ModelNotFittedError("Model must be fitted before making predictions.")
        average_attack, average_defense = self._trimmed_bottom_ratings(n_bottom=5, trim=2)
        if home_team not in self.team_ratings:
            attack_home = average_attack
            defense_home = average_defense
            self.team_ratings[home_team] = {"attack": attack_home, "defense": defense_home}
        else:
            attack_home = self.team_ratings[home_team]["attack"]
            defense_home = self.team_ratings[home_team]["defense"]

        if away_team not in self.team_ratings:
            attack_away = average_attack
            defense_away = average_defense
            self.team_ratings[away_team] = {"attack": attack_away, "defense": defense_away}
        else:
            attack_away = self.team_ratings[away_team]["attack"]
            defense_away = self.team_ratings[away_team]["defense"]


        lambda_home = np.exp(attack_home + defense_away + self.gamma)
        lambda_away = np.exp(attack_away + defense_home)
        prob_home, prob_draw, prob_away = self._calculate_match_probabilities(lambda_home, lambda_away, self.rho, max_goals=max_goals)

        logger.info("Predicted probabilities: %s, %s, %s", prob_home, prob_draw, prob_away)

        return prob_home, prob_draw, prob_away


    def _tau(self, home_goals: np.ndarray, away_goals: np.ndarray,
             lambda_home: np.ndarray, lambda_away: np.ndarray,
             rho: float) -> np.ndarray:
        """Compute the Dixon-Coles low-score adjustment factor.

        Applies the correlation correction tau(x, y) that adjusts
        the independent-Poisson probability for the four low-scoring
        outcomes (0-0, 1-0, 0-1, 1-1), where goal dependence between
        the two teams is strongest. All other scorelines have
        tau = 1.

        Parameters
        ----------
        home_goals : np.ndarray
            Home team's goals per match.
        away_goals : np.ndarray
            Away team's goals per match.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        rho : float
            Correlation parameter for the Dixon-Coles adjustment.

        Returns
        -------
        np.ndarray
            Tau adjustment factor for each match, floored at a small
            positive value (1e-10) to avoid taking the log of a
            non-positive number downstream.
        """
        tau = np.ones_like(home_goals, dtype=float)
        m = (home_goals == 0) & (away_goals == 0)
        tau[m] = 1 - lambda_home[m] * lambda_away[m] * rho

        m = (home_goals == 0) & (away_goals == 1)
        tau[m] = 1 + lambda_home[m] * rho

        m = (home_goals == 1) & (away_goals == 0)
        tau[m] = 1 + lambda_away[m] * rho

        m = (home_goals == 1) & (away_goals == 1)
        tau[m] = 1 - rho

        return np.maximum(tau, 1e-10)


    def _dtau_drho(self, home_goals: np.ndarray, 
                   away_goals: np.ndarray, lambda_home: np.ndarray, 
                   lambda_away: np.ndarray, tau: np.ndarray) -> float:
        """Gradient contribution of the tau term with respect to rho.

        For each match, computes d(log tau)/d(rho), restricted to
        the four low-score cells where tau differs from 1 (the
        derivative is 0 elsewhere).
        Subsequently, each contribution is weighted by the
        corresponding time-decay weight.
        The contributions are summed over all matches.

        Parameters
        ----------
        home_goals : np.ndarray
            Observed home goals per match.
        away_goals : np.ndarray
            Observed away goals per match.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        tau : np.ndarray
            Tau adjustment factor per match.

        Returns
        -------
        float
            Gradient of log(tau) with respect to rho.
        """
        d_tau_drho = np.zeros_like(home_goals, dtype=float)

        m = (home_goals == 0) & (away_goals == 0)
        d_tau_drho[m] = -lambda_home[m] * lambda_away[m] * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 0) & (away_goals == 1)
        d_tau_drho[m] = lambda_home[m] * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 1) & (away_goals == 0)
        d_tau_drho[m] = lambda_away[m] * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 1) & (away_goals == 1)
        d_tau_drho[m] = -self.delta_time_weights[m] / tau[m]

        return np.sum(d_tau_drho)


    def _dtau_dgamma(self, home_goals: np.ndarray, away_goals: np.ndarray, 
                     lambda_home: np.ndarray, lambda_away: np.ndarray, 
                     rho: float, tau: np.ndarray) -> float:
        """Gradient contribution of the tau term with respect to gamma.

        Only the (0,0) and (0,1) scoreline cells depend on gamma
        (which enters only through lambda_home), so the derivative
        is 0 elsewhere. Contributions are weighted by the time-decay
        weights and summed over all matches.

        Parameters
        ----------
        home_goals : np.ndarray
            Observed home goals per match.
        away_goals : np.ndarray
            Observed away goals per match.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        rho : float
            Correlation parameter.
        tau : np.ndarray
            Tau adjustment factor per match.

        Returns
        -------
        float
            Gradient of log(tau) with respect to gamma.
        """
        d_tau_dgamma = np.zeros_like(home_goals, dtype=float)

        m = (home_goals == 0) & (away_goals == 0)
        d_tau_dgamma[m] = -lambda_home[m] * lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 0) & (away_goals == 1)
        d_tau_dgamma[m] = lambda_home[m] * rho * self.delta_time_weights[m] / tau[m]

        return np.sum(d_tau_dgamma)


    def _dtau_dalpha(self, team: str, lambda_home: np.ndarray, lambda_away: np.ndarray,
                     home_goals: np.ndarray, away_goals: np.ndarray, rho: float,
                     tau: np.ndarray) -> float:
        """Gradient contribution of the tau term with respect to a team's attack rating.

        Sums d(log tau)/d(alpha_team) over every match involving
        said team (as either the home or away side), restricted to the
        low-score cells where tau != 1, weighted by the time-decay
        weights.

        Parameters
        ----------
        team : str
            Team name.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        home_goals : np.ndarray
            Observed home goals per match.
        away_goals : np.ndarray
            Observed away goals per match.
        rho : float
            Correlation parameter.
        tau : np.ndarray
            Tau adjustment factor per match.

        Returns
        -------
        float
            Time-weighted sum of d(log tau)/d(alpha_team) over all
            matches involving said team.
        """
        d_tau_dalpha = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self._make_team_mask(team)

        m = (home_goals == 0) & (away_goals == 0) & (home_mask | away_mask)
        d_tau_dalpha[m] = -lambda_home[m] * lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]   

        m = (home_goals == 0) & (away_goals == 1) & home_mask
        d_tau_dalpha[m] = lambda_home[m] * rho * self.delta_time_weights[m] / tau[m]  

        m = (home_goals == 1) & (away_goals == 0) & away_mask
        d_tau_dalpha[m] = lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]  

        return np.sum(d_tau_dalpha)


    def _dtau_dbeta(self, team: str, 
                    lambda_home: np.ndarray, lambda_away: np.ndarray,
                    home_goals: np.ndarray, away_goals: np.ndarray, 
                    rho: float, tau: np.ndarray) -> float:
        """
        Gradient contribution of the tau term with respect to a team's defense rating.

        Sums d(log tau)/d(beta_team) over every match involving
        said team (as either the home or away side), restricted to the
        low-score cells where tau != 1, weighted by the time-decay
        weights.

        Parameters
        ----------
        team : str
            Team name.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        home_goals : np.ndarray
            Observed home goals per match.
        away_goals : np.ndarray
            Observed away goals per match.
        rho : float
            Correlation parameter.
        tau : np.ndarray
            Tau adjustment factor per match.

        Returns
        -------
        float
            Time-weighted sum of d(log tau)/d(beta_team) over all
            matches involving said team.
        """
        d_tau_dbeta = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self._make_team_mask(team)

        m = (home_goals == 0) & (away_goals == 0) & (home_mask | away_mask)
        d_tau_dbeta[m] = -lambda_home[m] * lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]  

        m = (home_goals == 0) & (away_goals == 1) & away_mask
        d_tau_dbeta[m] = lambda_home[m] * rho * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 1) & (away_goals == 0) & home_mask
        d_tau_dbeta[m] = lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]

        return np.sum(d_tau_dbeta)


    def _dlogpoiss_dalpha(self, team: str, 
                          lambda_home: np.ndarray, 
                          lambda_away: np.ndarray,
                          home_goals: np.ndarray, 
                          away_goals: np.ndarray) -> float:
        """Gradient contribution of the Poisson term with respect to a team's attack rating.

        Since lambda_home = exp(attack_home + defense_away + gamma)
        and lambda_away = exp(attack_away + defense_home),
        d(log-Poisson)/d(alpha_team) equals (goals - lambda) in
        whichever role 'team' supplies the attack term for that
        match (home goals when 'team' plays at home, away goals
        when 'team' plays away). Contributions are weighted by the
        time-decay weights and summed over all matches involving
        'team'.

        Parameters
        ----------
        team : str
            Team name.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        home_goals : np.ndarray
            Observed home goals per match.
        away_goals : np.ndarray
            Observed away goals per match.

        Returns
        -------
        float
            Gradient contribution of the Poisson term with respect to the specified team's attack rating.
        """
        d_logpoiss_dalpha = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self._make_team_mask(team)

        d_logpoiss_dalpha[home_mask] = (home_goals[home_mask] - lambda_home[home_mask]) * self.delta_time_weights[home_mask]
        d_logpoiss_dalpha[away_mask] = (away_goals[away_mask] - lambda_away[away_mask]) * self.delta_time_weights[away_mask]
        return np.sum(d_logpoiss_dalpha)


    def _dlogpoiss_dbeta(self, team: str, 
                         lambda_home: np.ndarray, 
                         lambda_away: np.ndarray,
                         home_goals: np.ndarray, away_goals: np.ndarray) -> float:
        """Gradient contribution of the Poisson term with respect to a team's defense rating.

        A team's defense parameter enters its opponent's expected
        goals (beta_team affects lambda_away when 'team' is at home,
        and lambda_home when 'team' is away), so the derivative
        equals (opponent_goals - opponent_lambda) for matches
        involving 'team', weighted by the time-decay weights and
        summed.

        Parameters
        ----------
        team : str
            Team name.
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        lambda_away : np.ndarray
            Expected away goals (Poisson rate, not xG) per match.
        home_goals : np.ndarray
            Observed home goals per match.
        away_goals : np.ndarray
            Observed away goals per match.

        Returns
        -------
        float
            Gradient contribution of the Poisson term with respect to the specified team's defense rating.
        """
        d_logpoiss_dbeta = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self._make_team_mask(team)

        d_logpoiss_dbeta[away_mask] = (home_goals[away_mask] - lambda_home[away_mask]) * self.delta_time_weights[away_mask]
        d_logpoiss_dbeta[home_mask] = (away_goals[home_mask] - lambda_away[home_mask]) * self.delta_time_weights[home_mask]
        return np.sum(d_logpoiss_dbeta)


    def _dlogpoiss_dgamma(self, lambda_home: np.ndarray, 
                          home_goals: np.ndarray) -> float:
        """Gradient contribution of the Poisson term with respect to gamma.

        Since gamma only enters lambda_home, d(log-Poisson)/d(gamma)
        equals (home_goals - lambda_home) for every match, weighted
        by the time-decay weights and summed.

        Parameters
        ----------
        lambda_home : np.ndarray
            Expected home goals (Poisson rate, not xG) per match.
        home_goals : np.ndarray
            Observed home goals per match.

        Returns
        -------
        float
            Gradient contribution of the Poisson term with respect to the gamma parameter.
        """
        d_logpoiss_dgamma = home_goals - lambda_home
        return np.sum(d_logpoiss_dgamma * self.delta_time_weights)


    def _log_likelihood(self, params: np.ndarray, 
                        data: pl.DataFrame) -> float:
        """Compute the negative time-weighted Dixon-Coles log-likelihood.

        Unpacks the flat parameter vector into attack, defense,
        gamma, and rho, derives (Poisson rate) expected goals for 
        each match, and evaluates the Dixon-Coles 
        log-likelihood (Poisson log-pmf plus the tau low-score correction).
        This is weighted by the Dixon Coles xi parameter.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector of length 2 * n_teams + 2:
            attack ratings, defense ratings, gamma, and rho, in that order.
        data : pl.DataFrame
            Match data.

        Returns
        -------
        float
            Negative time-weighted log-likelihood of the data
        """
        attack, defense, gamma, rho = self._unpack_dc_params(params, self.n_teams)
        attack_home, attack_away, defense_home, defense_away = self._map_team_params_to_matches(attack, defense, self.home_teams, self.away_teams, self.teams_indices)
        lambda_home = np.exp(attack_home + defense_away + gamma)
        lambda_away = np.exp(attack_away + defense_home)
        home_goals = data["home_goals"].to_numpy()
        away_goals = data["away_goals"].to_numpy()
        log_tau = np.log(self._tau(home_goals, away_goals, lambda_home, lambda_away, rho))
        log_likelihood_per_match = log_tau + poisson.logpmf(home_goals, lambda_home) + poisson.logpmf(away_goals, lambda_away)
        return -np.sum(self.delta_time_weights * log_likelihood_per_match)


    def _log_likelihood_gradient(self, params: np.ndarray, 
                                 data: pl.DataFrame) -> np.ndarray:
        """Compute the analytic gradient of the negative log-likelihood
         with respect to all parameters.

        Combinines each parameter's Poisson log-likelihood
        derivative with its tau-correction derivative, and is passed 
        for use as the 'jac' argument to scipy.optimize.minimize.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector of length 2 * n_teams + 2:
            attack ratings, defense ratings, gamma, and rho, in that order.
        data : pl.DataFrame
            Match data.

        Returns
        -------
        np.ndarray
            Analytic gradient of the negative log-likelihood with
            respect to all parameters. Has same shape as params.
        """
        attack, defense, gamma, rho = self._unpack_dc_params(params, self.n_teams)
        attack_home, attack_away, defense_home, defense_away = self._map_team_params_to_matches(attack, defense, self.home_teams, self.away_teams, self.teams_indices)
        lambda_home = np.exp(attack_home + defense_away + gamma)
        lambda_away = np.exp(attack_away + defense_home)
        home_goals = data["home_goals"].to_numpy()
        away_goals = data["away_goals"].to_numpy()
        tau = self._tau(home_goals, away_goals, lambda_home, lambda_away, rho)
        gradient = np.zeros(len(params))
        grad_rho = self._dtau_drho(home_goals, away_goals, lambda_home, lambda_away, tau)
        grad_gamma = self._dtau_dgamma(home_goals, away_goals, lambda_home, lambda_away, rho, tau) + self._dlogpoiss_dgamma(lambda_home, home_goals)
        for idx, team in enumerate(self.teams_indices):
            gradient[idx] = self._dtau_dalpha(team, lambda_home, lambda_away, home_goals, away_goals, rho, tau) + self._dlogpoiss_dalpha(team, lambda_home, lambda_away, home_goals, away_goals)
            gradient[idx + self.n_teams] = self._dtau_dbeta(team, lambda_home, lambda_away, home_goals, away_goals, rho, tau) + self._dlogpoiss_dbeta(team, lambda_home, lambda_away, home_goals, away_goals)
        gradient[-2] = grad_gamma
        gradient[-1] = grad_rho
        return -gradient


    def _optimize_parameters(self, data: pl.DataFrame, 
                             require_convergence: bool = True) -> np.ndarray:
        """Fit model parameters via constrained maximum-likelihood optimization.
        Initializes attack/defense ratings from a small random
        normal distribution, gamma at 0.1, and rho at
        self.rho_init, then minimizes the negative Dixon-Coles
        log-likelihood subject to the constraint (required for uniqueness) 
        that attack ratings sum to zero (for identifiability).
        Per-parameter bounds are also enforced for numerical stability.

        Parameters
        ----------
        data : pl.DataFrame
            Match data used to evaluate the log-likelihood and its gradient.
        require_convergence : bool, optional
            If True, raise an OptimizationError if the optimizer fails to converge.

        Returns
        -------
        np.ndarray
            Optimized flat parameter vector 
            (attack, defense, gamma,rho).
            Returned by scipy.optimize.minimize.
        """
        logger.info("Starting optimization of Dixon-Coles model parameters.")
        initial_params = np.random.rand(2 * self.n_teams + 2)  # attack, defense, home_advantage, rho
        initial_params[:self.n_teams] = np.random.normal(0, 0.1, self.n_teams)
        initial_params[self.n_teams:2 * self.n_teams] = np.random.normal(0, 0.1, self.n_teams)
        initial_params[2 * self.n_teams] = 0.1
        initial_params[-1] = self.rho_init
        constraint = {"type": "eq",
                      "fun": lambda params: np.sum(params[:self.n_teams])}
        bounds = ([(-1.5, 1.5)] * self.n_teams + 
                  [(-1.5, 1.5)] * self.n_teams + 
                  [(0, 0.5)] + 
                  [(self.rho_bound, 0)])
        
        result = minimize(self._log_likelihood, initial_params, args=(data), constraints=constraint,
                           options=self.fit_options, bounds=bounds, jac=self._log_likelihood_gradient)
        if not result.success:
            logger.warning("Optimization failed to converge: %s", result.message)
            if require_convergence:
                raise OptimizationError("Failed to converge.")
        return result.x 


    def save_model(self, file_path):
        # Save the model parameters to a file
        # This is a placeholder for the actual saving logic
        raise NotImplementedError("Saving model is not yet implemented")


    def load_model(self, file_path):
        # Load the model parameters from a file
        # This is a placeholder for the actual loading logic
        raise NotImplementedError("Loading model is not yet implemented")


    def _clean_raw_result(self, 
                          raw_result: np.ndarray) -> tuple[dict[str, dict[str, float]], float, float]:
        """Convert the optimizer's flat parameter vector into structured outputs.

        Parameters
        ----------
        raw_result : np.ndarray
            The optimizer's flat parameter vector.

        Returns
        -------
        tuple[dict[str, dict[str, float]], float, float]
            A dictionary mapping each team name to its
            {"attack": ..., "defense": ...} ratings, followed by
            the fitted gamma and rho values.
        """
        team_ratings = dict({})
        for idx, team_name in enumerate(self.teams_indices):
            team_ratings[team_name] = {"attack": float(raw_result[idx]), 
                                       "defense" : float(raw_result[idx + self.n_teams])}
        gamma = float(raw_result[-2])
        rho = float(raw_result[-1])
        return team_ratings, gamma, rho
    

    def show_results(self) -> None:
        """Print a formatted summary of the fitted model.

        Displays each team's attack and defense ratings in an
        aligned table, followed by the fitted home advantage (gamma)
        and correlation (rho) parameters.
        """
        teams = [t for t in self.team_ratings.keys()]

        name_width = max(len(t) for t in teams) + 2

        print("Model: Dixon-Coles")
        print(f"Number of teams: {len(teams)}")
        print()
        print(f"{'Team':<{name_width}}{'Attack':<12}{'Defense':<12}")
        print("-" * (name_width + 24))
        for team in teams:
            attack = self.team_ratings[team]["attack"]
            defense = self.team_ratings[team]["defense"]
            print(f"{team:<{name_width}}{attack:<12.3f}{defense:<12.3f}")
        print("-" * (name_width + 24))
        print(f"Gamma: {self.gamma:.3f}")
        print(f"Rho: {self.rho:.4f}")


    def _store_team_info(self, data: pl.DataFrame) -> None:
        """Cache team indexing and home/away arrays needed for fitting.

        Parameters
        ----------
        data : pl.DataFrame
            Match data containing home_team and away_team columns.
        """
        self.teams_indices = self._map_teams_to_indices(data)
        self.n_teams = len(self.teams_indices)
        self.home_teams = data["home_team"].to_numpy()
        self.away_teams = data["away_team"].to_numpy()


    def _make_team_mask(self, team: str) -> tuple[np.ndarray, np.ndarray]:
        """Build boolean masks identifying a team's home and away matches.

        Parameters
        ----------
        team : str
            The team for which to build masks.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Boolean masks for home and away matches respectively.
        """
        home_mask = (self.home_teams == team)
        away_mask = (self.away_teams == team)   
        return home_mask, away_mask


    def _compute_time_decay_weights(self, 
                                    reference_date: datetime | date,
                                    xi: float, 
                                    data: pl.DataFrame) -> None:
        """Compute and store per-match time-decay weights.

        Parameters
        ----------
        reference_date : datetime | date
            Date from which time-decay weights are computed.
        xi : float
            Decay rate; xi = 0 yields no time decay;
        data : pl.DataFrame
            Match data containing match_date column.
        """
        time_diff = (reference_date - data["match_date"]).dt.total_days()
        self.delta_time_weights = self._phi(xi, time_diff.to_numpy())


    def _trimmed_bottom_ratings(self, n_bottom: int = 5, 
                                trim: int = 2) -> tuple[float, float]:
        """_summary_

        Parameters
        ----------
        n_bottom : int, optional
            The number of bottom teams to consider, by default 5
        trim : int, optional
            The number of bottom teams to exclude from the average, by default 2

        Returns
        -------
        tuple[float, float]
            The average attack and defense ratings of the weakest teams
        """
        if n_bottom <= trim:
            raise InvalidParameterError("n_bottom must be greater than trim.")
        
        teams = {k: v for k, v in self.team_ratings.items()}

        weakest_attack = np.array([v["attack"] for _, v in sorted(teams.items(), key=lambda kv: kv[1]["attack"])[:n_bottom]])
        weakest_defense = np.array([v["defense"] for _, v in sorted(teams.items(), key=lambda kv: kv[1]["defense"], reverse=True)[:n_bottom]])

        avg_attack = float(weakest_attack[trim:].mean())
        avg_defense = float(weakest_defense[trim:].mean())

        return avg_attack, avg_defense


    def _calculate_match_probabilities(self, lambda_home: float,
                                      lambda_away: float, rho: float,
                                      max_goals: int = 10) -> tuple[float, float, float]:
        """Compute win/draw/loss probabilities from Dixon-Coles expected goals.

        Sums the Dixon-Coles joint probability over
        every scoreline in the [0, max_goals] x [0, max_goals]
        grid, aggregating into home win, draw, and away win totals,
        then normalizes them to account for the truncated goal
        range.

        Parameters
        ----------
        lambda_home : float
            Dixon-Coles expected goals for the home team.
        lambda_away : float
            Dixon-Coles expected goals for the away team.
        rho : float
            Correlation parameter.
        max_goals : int, optional
            Maximum number of goals to consider, by default 10

        Returns
        -------
        tuple[float, float, float]
            Normalized probabilities of a home win, 
            draw, and away win in that order.
        """
        prob_home = 0.0
        prob_away = 0.0
        prob_draw = 0.0
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                tau = self._tau(np.array([home_goals]), np.array([away_goals]),
                                   np.array([lambda_home]), np.array([lambda_away]), rho)[0]
                prob = poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away) * tau
                if home_goals > away_goals:
                    prob_home += prob
                elif away_goals > home_goals:
                    prob_away += prob
                else:
                    prob_draw += prob

        p_home, p_draw, p_away = self._normalize_probabilities(prob_home, prob_draw, prob_away)
        return p_home, p_draw, p_away


    @staticmethod
    def _phi(xi: float, time_diff: np.ndarray) -> np.ndarray:
        """Calculate the decay function based on the time difference.

        Parameters
        ----------
        xi : float
            Decay rate.
        time_diff : np.ndarray
            Time differences.

        Returns
        -------
        np.ndarray
            Decay weight for each match.
        """
        return np.exp(-xi * time_diff)  


    @staticmethod
    def _map_teams_to_indices(data: pl.DataFrame) -> dict[str, int]:
        """Map team names to indices for parameter estimation

        Parameters
        ----------
        data : pl.DataFrame
            Match data containing "home_team" and "away_team" columns.

        Returns
        -------
        dict[str, int]
            Mapping from team name to index, in alphabetically sorted order.
        """
        teams = pl.concat([data["home_team"], data["away_team"]]).unique()
        team_to_index = {team: idx for idx, team in enumerate(teams.sort())}
        return team_to_index


    @staticmethod
    def _map_team_params_to_matches(attack, defense, home_teams, away_teams, teams_indices):
        """Broadcast per-team parameters onto per-match arrays.

        Parameters
        ----------
        attack : np.ndarray
            Attack strengths for each team indexed by team_indices.
        defense : np.ndarray
            Defense strengths for each team indexed by team_indices.
        home_teams : np.ndarray
            Array of home team names.
        away_teams : np.ndarray
            Array of away team names.
        teams_indices : dict[str, int]
            Mapping from team name to its position in attack and defense.
        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            Per-match arrays: home team's attack rating, away team's
            attack rating, home team's defense rating, and away
            team's defense rating.        
        """
        attack_home = attack[[teams_indices[team] for team in home_teams]]
        defense_home = defense[[teams_indices[team] for team in home_teams]]
        attack_away = attack[[teams_indices[team] for team in away_teams]]
        defense_away = defense[[teams_indices[team] for team in away_teams]]
        return attack_home, attack_away, defense_home, defense_away


    @staticmethod
    def _unpack_dc_params(params: np.ndarray, 
                          n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Split a flat parameter vector into
        attack strengths, defense strengths, gamma, and rho respectively.

        Parameters
        ----------
        params : np.ndarray
            Flat parameter vector of length 2 * n_teams + 2,
            ordered as attack ratings, defense ratings, gamma, then
            rho.
        n_teams : int
            The number of teams the models fits on.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float, float]
            Unpacked parameters: attack strengths, defense strengths, gamma, and rho.
        """
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        gamma = params[2 * n_teams] 
        rho = params[-1]
        return attack, defense, gamma, rho


    @staticmethod
    def _normalize_probabilities(*probs: float) -> tuple[float, ...]:
        """Normalizes a set of probabilities to sum up to unity.

        Parameters
        ----------
        *probs : float
            One or more raw probability values to normalize together.

        Returns
        -------
        tuple[float, ...]
            Normalized probabilities.
        """
        total = sum(probs)
        logger.info("Normalizing factor=%f", total)
        if total < 0.99:
            logger.warning("Normalizing factor is low: %f", total)
        return tuple(p / total for p in probs)