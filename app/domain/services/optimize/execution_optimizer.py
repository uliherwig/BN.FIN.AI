import optuna
from .optuna_service import OptunaService


class ExecutionOptimizationService:

    def __init__(self, train_service, optuna_service: OptunaService, indicators, model_params, n_jobs=4):
        self.train_service = train_service
        self.optuna_service = optuna_service
        self.indicators = indicators
        self.model_params = model_params
        self.n_jobs = n_jobs

    def objective(self, trial):

        settings = {
            "asset": "SPY",
            "start_date": "2010-01-01",
            "end_date": "2024-12-31",

            "price_change_threshold": 0.0016,
            "long_threshold": trial.suggest_float("long_threshold", 0.0, 0.0005),
            "short_threshold": trial.suggest_float("short_threshold", 0.0, 0.0005),

            "tp": trial.suggest_float("tp", 0.005, 0.05),
            "sl": trial.suggest_float("sl", 0.005, 0.03),
        }


        self.model_params["model_type"] = "regression"
        result = self.train_service.get_train_results_by_test_settings(
            settings=settings,
            model_params=self.model_params,
            indicators=self.indicators
        )
        sharpe = float(result["sharpe_ratio"])
        maxdd = float(result["max_drawdown"])
        profit = float(result["profit"])

        return (sharpe)

    def optimize(self, n_trials=50):
        study = self.optuna_service.create_study(
            "lgb_exec_opt",
            ["maximize"]
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs)
        return study
