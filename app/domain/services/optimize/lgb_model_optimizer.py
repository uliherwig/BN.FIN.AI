import optuna
from typing import Dict, List
from app.domain.services.train.lgb_train_service import LgbTrainService
from .optuna_service import OptunaService
from app.domain.domain_config import DOMAIN_CONFIG


class LgbModelOptimizationService:

    def __init__(self, train_service: LgbTrainService, optuna_service: OptunaService, indicator_models, n_jobs=4):
        self.train_service = train_service
        self.optuna_service = optuna_service
        self.indicator_models = indicator_models
        self.n_jobs = n_jobs

    def _suggest_model_params(self, trial):


        # need to hard code model_type for now
        
        return {
            "model_type": "regression",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 256),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000)
        }

    def objective(self, trial):

        model_params = self._suggest_model_params(trial)
      
        settings = DOMAIN_CONFIG["DEFAULT_EXEC_SETTINGS"]

        result = self.train_service.get_train_results_by_test_settings(
            settings=settings,
            model_params=model_params,
            indicators=self.indicator_models
        )
        sharpe = float(result["sharpe_ratio"])
        maxdd = float(result["max_drawdown"])
        profit = float(result["profit"])

        return (sharpe)

    def optimize(self, n_trials=50):
        study = self.optuna_service.create_study("lgb_model_opt", ["maximize"])
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)
        return study
