from app.config import GLOBAL_CONFIG
from app.domain.domain_config import DOMAIN_CONFIG

import optuna
from optuna.storages.journal import JournalStorage, JournalRedisBackend
from app.domain.operations.indicator_factory import IndicatorFactory
from app.domain import *



class OptunaService:
    
    def __init__(self):
        REDIS_URL = GLOBAL_CONFIG["REDIS_URL"]

        backend = JournalRedisBackend(REDIS_URL)
        self.storage = JournalStorage(backend)
        
    # TODO: handle existing studies so far we just delete them
    # existing_study = optuna.load_study(study_name=study_name, storage=storage)    
    def create_study(self, study_name: str, directions, is_new: bool = True) -> optuna.study.Study: 
        
        
        if is_new:
            try:
                optuna.delete_study(storage=self.storage, study_name=study_name)
            except KeyError:        
                pass

        return optuna.create_study(
            study_name=study_name,
            directions=directions,
            storage=self.storage,
            load_if_exists=not is_new,
            sampler=optuna.samplers.TPESampler(seed=45)
        )
 
        
    @staticmethod
    def get_standard_indicator_params(indicators: list[IndicatorEnum]) -> any:
        result = []
        param_standards = DOMAIN_CONFIG["INDICATOR_PARAM_DEFAULTS"]
        for enum in indicators:
    
            # ensure each instance gets its own dict
            params = param_standards[enum.name].copy()
            im = IndicatorModel(strategyType=enum,
                                params=json.dumps(params),
                                feature=DOMAIN_CONFIG["INDICATOR_FEATURES"][enum.name] if enum.name in DOMAIN_CONFIG["INDICATOR_FEATURES"] else ""
            )
            
            result.append(im)
        return result 

    @staticmethod
    def get_indicator_param_ranges(selected_indicators: list[str]) -> any:
        indicator_param_ranges = []
        # Use the class-level INDICATOR_PARAM_RANGES
        for sel in selected_indicators:
            # Support both IndicatorEnum and string names
            if isinstance(sel, str):
                key = sel.upper()
            else:
                key = sel.name
            if key in DOMAIN_CONFIG["INDICATOR_PARAM_RANGES"]:
                indicator_param_ranges.append(
                    (key, getattr(IndicatorEnum, key), DOMAIN_CONFIG["INDICATOR_PARAM_RANGES"][key])
                )
            else:
                raise ValueError(f"Unknown indicator: {sel}")
        return indicator_param_ranges
    
    @staticmethod
    def get_features_by_indicators(selected_indicators: list[IndicatorEnum]) -> any:
        features = []
        
        all_features = DOMAIN_CONFIG["INDICATOR_FEATURES"]
        for enum in selected_indicators:
            if enum.name in all_features:
                features.append(all_features[enum.name])
        return features
    
    def _log_features_to_file(self, trial_number: int, selected_features: list, profit: float, maxdd: float, sharpe: float):

        print("Logging selected features for trial",
              trial_number, ":", selected_features)
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
