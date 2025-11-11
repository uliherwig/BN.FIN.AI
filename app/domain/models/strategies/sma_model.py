from pydantic import BaseModel

class SmaModel(BaseModel):
    SMA_short: int      
    SMA_long: int