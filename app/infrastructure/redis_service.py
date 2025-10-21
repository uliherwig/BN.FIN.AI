import redis
import json
import os
from typing import List, Dict, Any
from functools import lru_cache

class RedisService:
    
    def __init__(self, host: str = os.getenv("REDIS_HOST", "localhost"), port: int = int(os.getenv("REDIS_PORT", 6379)), db: int = 0):        
        self.client = redis.Redis(host=host, port=port, db=db)
        
    def set_value(self, key: str, value: str):
        self.client.set(key, value)

    def get_value(self, key: str):
        return self.client.get(key)

    def save_chunk(self, key: str, data: List[Dict[str, Any]]) -> None:
        json_data = json.dumps(data)
        self.client.set(key, json_data)

    def get_chunk(self, key: str) -> List[Dict[str, Any]]:
        raw = self.client.get(key)
        if raw:
            return json.loads(raw)
        return []

    def delete_key(self, key: str) -> None:
        self.client.delete(key)

    def get_keys_by_pattern(self, pattern: str) -> List[str]:
        return [key.decode("utf-8") for key in self.client.keys(pattern)]
    
      
    
@lru_cache
def get_redis_service() -> RedisService:
    return RedisService()
