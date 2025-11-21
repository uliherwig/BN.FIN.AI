import json
import optuna
from typing import Any, Dict, List, Tuple
from app.domain import *

from app.domain.services.train.lgb_train_service import LgbTrainService
from .optuna_service import OptunaService

import os
import datetime

class IndicatorOptimizationService:

    def __init__(self, train_service: LgbTrainService, optuna_service: OptunaService, selected_indicators: list[IndicatorEnum], n_jobs: int = 4):
        self.train_service = train_service
        self.optuna_service = optuna_service        
        self.n_jobs = n_jobs
        self.selected_indicators = selected_indicators  

    def _suggest_indicators(self, trial: optuna.Trial) -> List[IndicatorModel]:

        indicator_configs = self.optuna_service.get_indicator_param_ranges(
            selected_indicators=self.selected_indicators
        )
        indicators = []
        for prefix, enum, param_ranges in indicator_configs:
            params = {k: trial.suggest_int(k, *v) for k, v in param_ranges.items()}
            indicators.append(IndicatorModel(strategyType=enum, params=json.dumps(params)))
        return indicators

    def objective(self, trial: optuna.Trial):

        selected_indicators = []
        # indicators
        for indicator in self.selected_indicators:
            if trial.suggest_categorical(f"{indicator.name}", [True, False]):
                selected_indicators.append(indicator)
       
        if len(selected_indicators) < 3:
            return (0.0, 1.0)
        
        # convert selected_indicators to string for logging
        selected_indicators_str = [str(indicator.name) for indicator in selected_indicators]
        
        trial.set_user_attr("selected_indicators", selected_indicators_str)
     
        # 
        indicators = self._suggest_indicators(trial)

        settings = {
            "asset": "SPY",
            "start_date": "2010-01-01",
            "end_date": "2024-12-31",

            # fixed thresholds, TP/SL
            "price_change_threshold": 0.0016,
            "long_threshold": 0.05,
            "short_threshold": 0.001,
            "tp": 0.02,
            "sl": 0.01,
        }

        # fixed model params (safe defaults)
        model_params = {
            "model_type": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_estimators": 100,
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
        
        self._log_features_to_file(trial.number, selected_indicators_str, profit, maxdd, sharpe)

        return (sharpe, maxdd)

    def optimize(self, n_trials=50):
        study = create_study(
            study_name="lgb_indicator_opt",
            directions=["maximize", "minimize"]
        )

        study.optimize(self.objective, n_trials=n_trials,
                       n_jobs=self.n_jobs, gc_after_trial=True)

        return study
    
    def _log_features_to_file(self, trial_number: int, selected_features: list, profit: float, maxdd: float, sharpe: float):
      
        print("Logging selected features for trial", trial_number, ":", selected_features)
        log_entry = {
            "trial_number": trial_number,       
            "selected_features": selected_features,
            "profit": profit,
            "max_drawdown": maxdd,
            "sharpe_ratio": sharpe
        }
        log_file = "optuna_feature_selection.log"
        if not os.path.exists(log_file):
            with open(log_file, "w") as f:
                json.dump([], f)
        # Robustly load or reset the log file if corrupted
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = []
        except json.JSONDecodeError:
            logs = []
        logs.append(log_entry)
        with open(log_file, "w") as f:
            json.dump(logs, f, indent=4)
