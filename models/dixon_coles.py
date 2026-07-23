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

    def fit(self, data):
        # Fit the Dixon-Coles model to the provided data
        self.teams_indices = self.map_teams_to_indices(data)
        self.n_teams = len(self.teams_indices)
        raw_result = self.optimize_parameters(data)
        self.result = self.clean_raw_result(raw_result)


    def predict(self, home_team, away_team):
        # Predict the outcome of a match between home_team and away_team
        # This is a placeholder for the actual prediction logic
        pass


    def log_likelihood(self, params, data):
        # Calculate the log-likelihood of the model given the parameters and data
        attack, defense, home_advantage, rho = self.unpack_dc_params(params, self.n_teams)
        home_teams = data["home_team"]
        away_teams = data["away_team"]
        attack_home, attack_away, defense_home, defense_away = self.map_team_params_to_matches(attack, defense, home_teams, away_teams, self.teams_indices)
        lambda_home = np.exp(attack_home + defense_away + home_advantage)
        lambda_away = np.exp(attack_away + defense_home)
        log_tau = np.log(self.tau(data["home_goals"], data["away_goals"], lambda_home, lambda_away, rho))
        return -np.sum(log_tau + poisson.logpmf(data["home_goals"], lambda_home) + poisson.logpmf(data["away_goals"], lambda_away))

    
    def optimize_parameters(self, data):
        initial_params = np.random.rand(2 * self.n_teams + 2)  # attack, defense, home_advantage, rho
        initial_params[:self.n_teams] = np.random.normal(0, 0.1, self.n_teams)
        initial_params[self.n_teams:2 * self.n_teams] = np.random.normal(0, 0.1, self.n_teams)
        initial_params[2 * self.n_teams] = 0.1
        initial_params[-1] = 0 
        constraint = {"type": "eq",
                      "fun": lambda params: np.sum(params[:self.n_teams])}
        bounds = ([(-1.5, 1.5)] * self.n_teams +   # attack
                  [(-1.5, 1.5)] * self.n_teams +   # defense
                  [(0, 0.5)] +        # home_advantage
                  [(self.rho_bound, 0)]      # rho: we start with pure poisson, so rho=0 for the first stage
                  )
        
        result = minimize(self.log_likelihood, initial_params, args=(data), constraints=constraint, options=self.fit_options, bounds=bounds)
        return result.x 


    def save_model(self, file_path):
        # Save the model parameters to a file
        # This is a placeholder for the actual saving logic
        pass


    def load_model(self, file_path):
        # Load the model parameters from a file
        # This is a placeholder for the actual loading logic
        pass


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


    def clean_raw_result(self, raw_result):
        result = dict({})
        for idx, team_name in enumerate(self.teams_indices):
            result[team_name] = {"attack": float(raw_result[idx]), 
                                 "defense" : float(raw_result[idx + self.n_teams])}
        result["home advantage"] = float(raw_result[-2])
        result["rho"] = float(raw_result[-1])
        return result
    

    def show_results(self):
        teams = [t for t in self.result if t not in ("home advantage", "rho")]

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
        print(f"Home Advantage: {self.result['home advantage']:.3f}")
        print(f"Rho: {self.result['rho']:.4f}")


    @staticmethod
    def decay_function(xi, time_diff):
        # Calculate the decay function based on the time difference
        return np.exp(-xi * time_diff)  


    @staticmethod
    def map_teams_to_indices(data):
        # Map team names to indices for parameter estimation
        teams = data["home_team"].append(data["away_team"]).unique()
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
        home_advantage = params[2 * n_teams]
        rho = params[-1]
        return attack, defense, home_advantage, rho