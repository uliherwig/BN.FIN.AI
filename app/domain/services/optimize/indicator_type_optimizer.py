import json
import optuna
from typing import Any, Dict, List, Tuple
from app.domain import *

from app.domain.services.train.lgb_train_service import LgbTrainService
from .optuna_service import OptunaService
import os
import datetime

class IndicatorTypeOptimizationService:

    def __init__(self, train_service: LgbTrainService, optuna_service: OptunaService, n_jobs: int = 4):
        self.train_service = train_service
        self.n_jobs = n_jobs 
        self.optuna_service = optuna_service     
        
        # extract indicators from IndicatorEnum except None
        self.selected_indicators = [indicator for indicator in IndicatorEnum if indicator != IndicatorEnum.NONE and indicator != IndicatorEnum.BREAKOUT]



    def objective(self, trial: optuna.Trial):

        selected_indicators = []
        # indicators
        for indicator in self.selected_indicators:
            if trial.suggest_categorical(f"{indicator.name}", [True, False]):
                selected_indicators.append(indicator)
       
        if len(selected_indicators) < 3:
            return (0.0, 1.0) 
        
        
        # 
        indicators = self.optuna_service.get_standard_indicator_params(
            selected_indicators)

        settings = {
            "asset": "SPY",
            "start_date": "2010-01-01",
            "end_date": "2024-12-31",

            # fixed thresholds, TP/SL
            "price_change_threshold": 0.0016,
            "long_threshold": 0.0003,
            "short_threshold": 0.0002,
            "tp": 0.021,
            "sl": 0.012,
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
        
        result = self.train_service.get_train_results_by_test_settings(
            settings=settings,
            model_params=model_params,
            indicators=indicators
        )

        sharpe = float(result["sharpe_ratio"])
        maxdd = float(result["max_drawdown"]) 
        profit = float(result["profit"])
        
        # self._log_features_to_file(trial.number, selected_indicators_str, profit, maxdd, sharpe)

        return (sharpe)

    def optimize(self, n_trials=20):
        study = self.optuna_service.create_study(
            study_name="lgb_indicator_type_opt",
            directions=["maximize"]
           
        )

        study.optimize(self.objective, n_trials=n_trials,
                       n_jobs=self.n_jobs, gc_after_trial=True)

        return study
    

