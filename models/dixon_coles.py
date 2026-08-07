from scipy.optimize import minimize
import numpy as np
import polars as pl
from scipy.stats import poisson
np.seterr(invalid="raise")

class DixonColes:
    def __init__(self, fit_options=None, data=None, rho_init=None, rho_bound=None):
        self.data = None
        if fit_options is None:
            self.fit_options = {"disp": True, "maxiter": 1000}
        else:
            self.fit_options = fit_options

        if rho_init == None:
            self.rho_init = -0.05
        else:
            self.rho_init = rho_init

        if rho_bound == None:
            self.rho_bound = -0.5
        else:
            self.rho_bound = rho_bound


    def fit(self, data, reference_date=None, xi=0):
        # Fit the Dixon-Coles model to the provided data
        self.store_team_info(data)
        if reference_date == None:
            reference_date = max(data["match_date"])
        self.compute_time_decay_weights(reference_date, xi, data)
        # self._validate_gradient_func(data)
        raw_result = self.optimize_parameters(data)
        self.result = self.clean_raw_result(raw_result)


    def predict(self, home_team, away_team, max_goals=10, verbose=False):
        # Predict the outcome of a match between home_team and away_team
        prob_home = 0
        prob_draw = 0
        prob_away = 0
        attack_home = self.result[home_team]["attack"]
        attack_away = self.result[away_team]["attack"]
        defense_home = self.result[home_team]["defense"]
        defense_away = self.result[away_team]["defense"]
        gamma = self.result["gamma"]
        rho = self.result["rho"]
        lambda_home = np.exp(attack_home + defense_away + gamma)
        lambda_away = np.exp(attack_away + defense_home)
        for home_goals in range(max_goals):
            for away_goals in range(max_goals):
                tau = self.tau(np.array([home_goals]), np.array([away_goals]),
                                   np.array([lambda_home]), np.array([lambda_away]), rho)[0]
                prob = poisson.pmf(home_goals, lambda_home) * poisson.pmf(away_goals, lambda_away) * tau
                if home_goals > away_goals:
                    prob_home += prob
                elif away_goals > home_goals:
                    prob_away += prob
                else:
                    prob_draw += prob

        normalize_factor = prob_home + prob_away + prob_draw
        if normalize_factor < 0.99:
            print("")
        prob_home /= normalize_factor
        prob_away /= normalize_factor
        prob_draw /= normalize_factor

        if verbose:
            print(prob_home, prob_draw, prob_away)
            print(normalize_factor)
            print(prob_home + prob_draw + prob_away)

        return prob_home, prob_draw, prob_away


    def tau(self, home_goals, away_goals, lambda_home, lambda_away, rho):
        # Calculate the tau function for the Dixon-Coles model
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


    def dtau_drho(self, home_goals, away_goals, lambda_home, lambda_away, tau):
        """Returns dtau/drho same masked structure as tau() and dtau_dlambda()."""
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


    def dtau_dgamma(self, home_goals, away_goals, lambda_home, lambda_away, rho, tau):
        """Returns dtau/dgamma same masked structure as tau() and dtau_dlambda()."""
        d_tau_dgamma = np.zeros_like(home_goals, dtype=float)

        m = (home_goals == 0) & (away_goals == 0)
        d_tau_dgamma[m] = -lambda_home[m] * lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 0) & (away_goals == 1)
        d_tau_dgamma[m] = lambda_home[m] * rho * self.delta_time_weights[m] / tau[m]

        return np.sum(d_tau_dgamma)


    def dtau_dalpha(self, team, lambda_home, lambda_away,
                     home_goals, away_goals, rho, tau):
        d_tau_dalpha = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self.make_team_mask(team)

        m = (home_goals == 0) & (away_goals == 0) & (home_mask | away_mask)
        d_tau_dalpha[m] = -lambda_home[m] * lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]   

        m = (home_goals == 0) & (away_goals == 1) & home_mask
        d_tau_dalpha[m] = lambda_home[m] * rho * self.delta_time_weights[m] / tau[m]  

        m = (home_goals == 1) & (away_goals == 0) & away_mask
        d_tau_dalpha[m] = lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]  

        return np.sum(d_tau_dalpha)


    def dtau_dbeta(self, team, lambda_home, lambda_away,
                         home_goals, away_goals, rho, tau):
        d_tau_dbeta = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self.make_team_mask(team)

        m = (home_goals == 0) & (away_goals == 0) & (home_mask | away_mask)
        d_tau_dbeta[m] = -lambda_home[m] * lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]  

        m = (home_goals == 0) & (away_goals == 1) & away_mask
        d_tau_dbeta[m] = lambda_home[m] * rho * self.delta_time_weights[m] / tau[m]

        m = (home_goals == 1) & (away_goals == 0) & home_mask
        d_tau_dbeta[m] = lambda_away[m] * rho * self.delta_time_weights[m] / tau[m]

        return np.sum(d_tau_dbeta)


    def dlogpoiss_dalpha(self, team, lambda_home, lambda_away,
                     home_goals, away_goals):
        d_logpoiss_dalpha = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self.make_team_mask(team)

        d_logpoiss_dalpha[home_mask] = (home_goals[home_mask] - lambda_home[home_mask]) * self.delta_time_weights[home_mask]
        d_logpoiss_dalpha[away_mask] = (away_goals[away_mask] - lambda_away[away_mask]) * self.delta_time_weights[away_mask]
        return np.sum(d_logpoiss_dalpha)


    def dlogpoiss_dbeta(self, team, lambda_home, lambda_away,
                     home_goals, away_goals):
        d_logpoiss_dbeta = np.zeros_like(home_goals, dtype=float)
        home_mask, away_mask = self.make_team_mask(team)

        d_logpoiss_dbeta[away_mask] = (home_goals[away_mask] - lambda_home[away_mask]) * self.delta_time_weights[away_mask]
        d_logpoiss_dbeta[home_mask] = (away_goals[home_mask] - lambda_away[home_mask]) * self.delta_time_weights[home_mask]
        return np.sum(d_logpoiss_dbeta)


    def dlogpoiss_dgamma(self, lambda_home, home_goals):
        d_logpoiss_dgamma = home_goals - lambda_home
        return np.sum(d_logpoiss_dgamma * self.delta_time_weights)


    def log_likelihood(self, params, data):
        # Calculate the log-likelihood of the model given the parameters and data
        attack, defense, gamma, rho = self.unpack_dc_params(params, self.n_teams)
        attack_home, attack_away, defense_home, defense_away = self.map_team_params_to_matches(attack, defense, self.home_teams, self.away_teams, self.teams_indices)
        lambda_home = np.exp(attack_home + defense_away + gamma)
        lambda_away = np.exp(attack_away + defense_home)
        log_tau = np.log(self.tau(data["home_goals"], data["away_goals"], lambda_home, lambda_away, rho))
        log_likelihood_per_match = log_tau + poisson.logpmf(data["home_goals"], lambda_home) + poisson.logpmf(data["away_goals"], lambda_away)
        return -np.sum(self.delta_time_weights * log_likelihood_per_match)


    def log_likelihood_gradient(self, params, data):
        attack, defense, gamma, rho = self.unpack_dc_params(params, self.n_teams)
        attack_home, attack_away, defense_home, defense_away = self.map_team_params_to_matches(attack, defense, self.home_teams, self.away_teams, self.teams_indices)
        lambda_home = np.exp(attack_home + defense_away + gamma)
        lambda_away = np.exp(attack_away + defense_home)
        home_goals = data["home_goals"].to_numpy()
        away_goals = data["away_goals"].to_numpy()
        tau = self.tau(home_goals, away_goals, lambda_home, lambda_away, rho)
        gradient = np.zeros(len(params))
        grad_rho = self.dtau_drho(home_goals, away_goals, lambda_home, lambda_away, tau)
        grad_gamma = self.dtau_dgamma(home_goals, away_goals, lambda_home, lambda_away, rho, tau) + self.dlogpoiss_dgamma(lambda_home, home_goals)
        for idx, team in enumerate(self.teams_indices):
            gradient[idx] = self.dtau_dalpha(team, lambda_home, lambda_away, home_goals, away_goals, rho, tau) + self.dlogpoiss_dalpha(team, lambda_home, lambda_away, home_goals, away_goals)
            gradient[idx + self.n_teams] = self.dtau_dbeta(team, lambda_home, lambda_away, home_goals, away_goals, rho, tau) + self.dlogpoiss_dbeta(team, lambda_home, lambda_away, home_goals, away_goals)
        gradient[-2] = grad_gamma
        gradient[-1] = grad_rho
        return -gradient


    def optimize_parameters(self, data):
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
        
        result = minimize(self.log_likelihood, initial_params, args=(data), constraints=constraint,
                           options=self.fit_options, bounds=bounds, jac=self.log_likelihood_gradient)
        return result.x 


    def save_model(self, file_path):
        # Save the model parameters to a file
        # This is a placeholder for the actual saving logic
        pass


    def load_model(self, file_path):
        # Load the model parameters from a file
        # This is a placeholder for the actual loading logic
        pass


    def clean_raw_result(self, raw_result):
        result = dict({})
        for idx, team_name in enumerate(self.teams_indices):
            result[team_name] = {"attack": float(raw_result[idx]), 
                                 "defense" : float(raw_result[idx + self.n_teams])}
        result["gamma"] = float(raw_result[-2])
        result["rho"] = float(raw_result[-1])
        return result
    

    def show_results(self):
        teams = [t for t in self.result if t not in ("gamma", "rho")]

        name_width = max(len(t) for t in teams) + 2

        print("Model: Dixon-Coles")
        print(f"Number of teams: {len(teams)}")
        print()
        print(f"{'Team':<{name_width}}{'Attack':<12}{'Defense':<12}")
        print("-" * (name_width + 24))
        for team in teams:
            attack = self.result[team]["attack"]
            defense = self.result[team]["defense"]
            print(f"{team:<{name_width}}{attack:<12.3f}{defense:<12.3f}")
        print("-" * (name_width + 24))
        print(f"Gamma: {self.result['gamma']:.3f}")
        print(f"Rho: {self.result['rho']:.4f}")


    def store_team_info(self, data):
        self.teams_indices = self.map_teams_to_indices(data)
        self.n_teams = len(self.teams_indices)
        self.home_teams = data["home_team"].to_numpy()
        self.away_teams = data["away_team"].to_numpy()


    def make_team_mask(self, team):
            home_mask = (self.home_teams == team)
            away_mask = (self.away_teams == team)   
            return home_mask, away_mask


    def compute_time_decay_weights(self, reference_date, xi, data):
        time_diff = (reference_date - data["match_date"]).dt.total_days()
        self.delta_time_weights = self.phi(xi, time_diff.to_numpy())


    @staticmethod
    def phi(xi, time_diff):
        # Calculate the decay function based on the time difference
        return np.exp(-xi * time_diff)  


    @staticmethod
    def map_teams_to_indices(data):
        # Map team names to indices for parameter estimation
        teams = pl.concat([data["home_team"], data["away_team"]]).unique()
        team_to_index = {team: idx for idx, team in enumerate(teams.sort())}
        return team_to_index


    @staticmethod
    def map_team_params_to_matches(attack, defense, home_teams, away_teams, teams_indices):
        attack_home = attack[[teams_indices[team] for team in home_teams]]
        defense_home = defense[[teams_indices[team] for team in home_teams]]
        attack_away = attack[[teams_indices[team] for team in away_teams]]
        defense_away = defense[[teams_indices[team] for team in away_teams]]
        return attack_home, attack_away, defense_home, defense_away


    @staticmethod
    def unpack_dc_params(params, n_teams):
        attack = params[:n_teams]
        defense = params[n_teams:2 * n_teams]
        gamma = params[2 * n_teams] 
        rho = params[-1]
        return attack, defense, gamma, rho
