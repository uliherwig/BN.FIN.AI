import optuna
from typing import Dict, List
from app.domain.services.train.lgb_train_service import LgbTrainService
from .shared_optuna import create_study


class ModelOptimizationService:

    def __init__(self, train_service: LgbTrainService, indicators, n_jobs=4):
        self.train_service = train_service
        self.indicators = indicators
        self.n_jobs = n_jobs

    def _suggest_model_params(self, trial):

        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 150, 600)
        }

    def objective(self, trial):

        model_params = self._suggest_model_params(trial)

        settings = {
            "asset": "SPY",
            "start_date": "2010-01-01",
            "end_date": "2024-12-31",

            "price_change_threshold": 0.0016,
            "long_threshold": 0.7,
            "short_threshold": 0.7,
            "tp": 0.04,
            "sl": 0.01
        }

        result = self.train_service.get_train_results_by_test_settings(
            settings=settings,
            model_params=model_params,
            indicators=self.indicators
        )
        sharpe = float(result["sharpe_ratio"])
        maxdd = float(result["max_drawdown"])
        profit = float(result["profit"])

        return (sharpe, maxdd)

    def optimize(self, n_trials=50):
        study = create_study("lgb_model_opt", ["maximize", "minimize"])
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)
        return study
