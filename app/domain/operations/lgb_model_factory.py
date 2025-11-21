import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score
import os
import joblib

class LGBModelFactory:
    
    @staticmethod
    def create_lgb_model(params: dict = None) :
        
        
        # print(f"Creating LightGBM model of type: {model_type} with params: {params}")
        model_type = params.get("model_type")
        learning_rate = params.get("learning_rate")
        num_leaves = params.get("num_leaves")
        max_depth = params.get("max_depth")
        subsample = params.get("subsample")
        colsample_bytree = params.get("colsample_bytree")
        n_estimators = params.get("n_estimators")
        

        if params is None:
            params = {}
        if model_type == 'classification':
            
            return lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=20,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                class_weight='balanced',  # Important for imbalanced data
                random_state=42,
                verbose=-1     
            )
        elif model_type == 'regression':
            
            return lgb.LGBMRegressor(
                objective='regression',
                metric="rmse",
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                num_leaves=num_leaves,
                min_child_samples=20,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=42,
                verbose=-1,
               
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    @staticmethod
    def save_lgb_model(model, asset, features, scaler=None) -> None:
        os.makedirs('ai_models', exist_ok=True)
        day = pd.Timestamp.now().strftime("%Y%m%d")
        model_path = f'ai_models/lgb_{asset}_{day}.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Model saved to {model_path}")
        if scaler is not None:
            scaler_path = f'ai_models/lgb_{asset}_{day}_scaler.pkl'
            joblib.dump(scaler, scaler_path)
            print(f"✅ Scaler saved to {scaler_path}")
        features_path = f'ai_models/lgb_{asset}_{day}_features.pkl'   
        joblib.dump(features, features_path)
        print(f"✅ Features saved to {features_path}")

    @staticmethod
    def load_lgb_model(asset: str, day: str):
        try:
            if day is None:
                day = pd.Timestamp.now().strftime("%Y%m%d")
            model_path = f'ai_models/lgb_{asset}_{day}.pkl'
            features_path = f'ai_models/lgb_{asset}_{day}_features.pkl'
            scaler_path = f'ai_models/lgb_{asset}_{day}_scaler.pkl'
            model = joblib.load(model_path)
            print(f"✅ Model loaded from {model_path}")
            features = joblib.load(features_path)
            print(f"✅ Features loaded: {features}")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"✅ Scaler loaded from {scaler_path}")
                return model, features, scaler
            else:
                return model, features
        except FileNotFoundError as e:
            print(f"❌ Model not found: {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
