import xgboost as xgb
import numpy as np
import polars as pl
from exceptions import ModelNotFittedError
import logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "enable_categorical": True,
    "tree_method": "hist",
    "n_jobs": 8,
    "max_depth": 4,
    "eta": 0.03,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.65,
    "gamma": 2,
    "reg_alpha": 0.5,
    "reg_lambda": 10,
    "max_delta_step": 3}

class XGBoostModel:
    def __init__(self, feature_columns: list[str],
                 target_col: str = "outcome",
                params: dict | None = None) -> None:
        """_summary_

        Parameters
        ----------
        feature_columns : list[str]
            _description_
        target_col : str, optional
            _description_, by default "outcome"
        params : dict | None, optional
            _description_, by default None
        """
        merged_params = {**DEFAULT_PARAMS, **(params or {})}
        self._model = xgb.XGBClassifier(**merged_params)
        self.feature_columns = feature_columns
        self.target_col = target_col


    def fit(self, data: pl.DataFrame, **kwargs) -> None:
        """Wrapper around xgb.XGBClassifier.fit().

        Parameters
        ----------
        data : pl.DataFrame
            Data to fit to.
        """
        X_selected = data.select(self.feature_columns)
        self.schema = X_selected.schema
        y = data[self.target_col]
        logger.info("Fitting XGBoost model on %d rows, %d features.", *X_selected.shape)
        self._model.fit(X_selected, y, **kwargs)
        self._fitted = True


    def predict(self, X: pl.DataFrame) -> np.ndarray:
        """Wrapper arouond xgb.XGBClassifier.predict_proba().

        Parameters
        ----------
        X : pl.DataFrame
            Data to predict on.

        Returns
        -------
        np.ndarray
            Array (or array of array) of probabilities.

        Raises
        ------
        ModelNotFittedError
            _description_
        """
        if not hasattr(self, "_fitted"):
            raise ModelNotFittedError("Model must be fitted before predicting.")
        X_features = X.select(self.feature_columns).cast(self.schema)
        return self._model.predict_proba(X_features)


    @property
    def feature_importances_(self) -> np.ndarray:
        """_summary_

        Returns
        -------
        np.ndarray
            _description_

        Raises
        ------
        ModelNotFittedError
            _description_
        """
        if not hasattr(self, "_fitted"):
            raise ModelNotFittedError("Model must be fitted before accessing feature_importances_.")
        return self._model.feature_importances_


    def show_results(self) -> None:
        """Prints out the list of features and their importance values.
        """
        importance = (pl.DataFrame({"feature": self.feature_columns, 
                                    "importance": self.feature_importances_})
                                    .sort("importance", descending=True))
        name_width = max(len(f) for f in importance["feature"]) + 2
        print("Model: XGBoost")
        print(f"{'Feature':<{name_width}}Importance")
        print("-" * (name_width + 12))
        for feat, imp in zip(importance["feature"], importance["importance"]):
            print(f"{feat:<{name_width}}{imp:.4f}")


    def save_model(self, file_path: str) -> None:
        """_summary_

        Parameters
        ----------
        file_path : str
            _description_
        """
        self._model.save_model(file_path)  


    def load_model(self, file_path: str) -> None:
        """_summary_

        Parameters
        ----------
        file_path : str
            _description_
        """
        self._model.load_model(file_path)