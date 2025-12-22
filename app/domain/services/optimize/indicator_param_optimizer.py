import json
import optuna
from typing import Any, Dict, List, Tuple
from app.domain.domain_config import DOMAIN_CONFIG
from app.domain import *

from app.domain.services.train.lgb_train_service import LgbTrainService
from .optuna_service import OptunaService
import os
import datetime

class IndicatorParamOptimizationService:

    def __init__(self, train_service: LgbTrainService, optuna_service: OptunaService, selected_indicators: list[str], n_jobs: int = 4):
        self.train_service = train_service
        self.optuna_service = optuna_service
        self.n_jobs = n_jobs
        self.selected_indicators = selected_indicators  

    def _suggest_indicators(self, trial: optuna.Trial) -> List[IndicatorModel]:

        indicator_param_ranges = self.optuna_service.get_indicator_param_ranges(
            selected_indicators=self.selected_indicators
        )
        indicator_models = []
        for prefix, enum, param_ranges in indicator_param_ranges:
            # print(f"Suggesting params for indicator: {prefix} with ranges: {param_ranges} enum: {enum}")
            params = {k: trial.suggest_int(k, *v) for k, v in param_ranges.items()}
            im = IndicatorModel(strategyType=enum, 
                                params=json.dumps(params),
                                feature=DOMAIN_CONFIG["INDICATOR_FEATURES"][prefix])
            
            indicator_models.append(im)
        return indicator_models
    
    def objective(self, trial: optuna.Trial):
     
        indicator_models = self._suggest_indicators(trial)  
        
        result = self.train_service.get_train_results_by_test_settings(
            settings=DOMAIN_CONFIG["DEFAULT_EXEC_SETTINGS"],
            model_params=DOMAIN_CONFIG["DEFAULT_MODEL_PARAMS"],
            indicators=indicator_models
        )

        sharpe = float(result["sharpe_ratio"])
        maxdd = float(result["max_drawdown"]) 
        profit = float(result["profit"])
        
        return (sharpe)

    def optimize(self, n_trials=50):
        study = self.optuna_service.create_study(
            study_name="lgb_indicator_params_opt",
            directions=["maximize"]
        )

        study.optimize(self.objective, n_trials=n_trials,
                       n_jobs=self.n_jobs, gc_after_trial=True)

        return study


