import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, r2_score
import os
import joblib
import redis
import pickle
from app.infrastructure.redis_service import RedisService, get_redis_service


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
    def save_lgb_model(model, asset, period, features, scaler=None) -> None:
        """
        Save model, features, and scaler to Redis using asset and period in the key.
        """
        redis_service = get_redis_service()

        base_key = f"lgb:{asset}:{period}"
        # Save model
        
        redis_service.set_value(f"{base_key}:model", pickle.dumps(model))
        print(f"✅ Model saved to Redis key {base_key}:model")
        # Save features
        redis_service.set_value(f"{base_key}:features", pickle.dumps(features))
        print(f"✅ Features saved to Redis key {base_key}:features")
        # Save scaler if present
        if scaler is not None:
            redis_service.set_value(f"{base_key}:scaler", pickle.dumps(scaler))
            print(f"✅ Scaler saved to Redis key {base_key}:scaler")

    @staticmethod
    def load_lgb_model(asset: str, period: str):
        """
        Load model, features, and scaler from Redis using asset and period in the key.
        """
        redis_service = get_redis_service() 
        base_key = f"lgb:{asset}:{period}"
        try:
            model_bytes = redis_service.get_value(f"{base_key}:model")
            features_bytes = redis_service.get_value(f"{base_key}:features")
            scaler_bytes = redis_service.get_value(f"{base_key}:scaler")
            if model_bytes is None or features_bytes is None:
                print(f"❌ Model or features not found in Redis for {base_key}")
                return False
            model = pickle.loads(model_bytes)
            print(f"✅ Model loaded from Redis key {base_key}:model")
            features = pickle.loads(features_bytes)
            print(f"✅ Features loaded from Redis key {base_key}:features")
            if scaler_bytes is not None:
                scaler = pickle.loads(scaler_bytes)
                print(f"✅ Scaler loaded from Redis key {base_key}:scaler")
                return model, features, scaler
            else:
                return model, features
        except Exception as e:
            print(f"❌ Error loading model from Redis: {e}")
            return False
