import json
import optuna
from typing import Any, Dict, List, Tuple
from app.domain.services.train.lgb_train_service import LgbTrainService
from app.domain.models.enums import IndicatorEnum
from app.domain.models.strategies.indicator_model import IndicatorModel
from .shared_optuna import create_study
from .optuna_configurator import OptunaConfigurator


class IndicatorOptimizationService:

    def __init__(self, train_service: LgbTrainService, selected_indicators: list[IndicatorEnum], n_jobs: int = 4):
        self.train_service = train_service
        self.n_jobs = n_jobs
        self.selected_indicators = selected_indicators  

    def _suggest_indicators(self, trial: optuna.Trial) -> List[IndicatorModel]:

        indicator_configs = OptunaConfigurator.get_indicator_config(
            selected_indicators=self.selected_indicators
        )
        indicators = []
        for prefix, enum, param_ranges in indicator_configs:
            params = {k: trial.suggest_int(k, *v) for k, v in param_ranges.items()}
            indicators.append(IndicatorModel(strategyType=enum, params=json.dumps(params)))
        return indicators

    def objective(self, trial: optuna.Trial):

        indicators = self._suggest_indicators(trial)

        settings = {
            "asset": "SPY",
            "start_date": "2010-01-01",
            "end_date": "2024-12-31",

            # fixed thresholds, TP/SL
            "price_change_threshold": 0.0016,
            "long_threshold": 0.7,
            "short_threshold": 0.7,
            "tp": 0.02,
            "sl": 0.01,
        }

        # fixed model params (safe defaults)
        model_params = {
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_estimators": 500,
        }
        
        # model = lgb.LGBMClassifier(
        #     n_estimators=500,
        #     learning_rate=0.05,
        #     max_depth=5,
        #     num_leaves=31,
        #     min_child_samples=20,
        #     subsample=0.8,
        #     colsample_bytree=0.8,
        #     class_weight='balanced',  # Important for imbalanced data
        #     random_state=42,
        #     verbose=-1
        # )

        result = self.train_service.get_train_results_by_test_settings(
            settings=settings,
            model_params=model_params,
            indicators=indicators
        )

        sharpe = float(result["sharpe_ratio"])
        maxdd = float(result["max_drawdown"]) 
        profit = float(result["profit"])

        return (sharpe, maxdd)

    def optimize(self, n_trials=50):
        study = create_study(
            study_name="lgb_indicator_opt",
            directions=["maximize", "minimize"]
        )

        study.optimize(self.objective, n_trials=n_trials,
                       n_jobs=self.n_jobs, gc_after_trial=True)

        return study
