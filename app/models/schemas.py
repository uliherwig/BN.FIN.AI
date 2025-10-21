from pydantic import BaseModel
from datetime import datetime


# ...existing code...

class QouteData(BaseModel):
    stamp: datetime  # UTC-Zeitstempel, z.B. "2024-06-11T12:34:56Z"
    bid: float
    ask: float

class InputData(BaseModel):
    name: str
    value: int

class Result(BaseModel):
    message: str
    
class StrategyData(BaseModel):
    Id: str
    Name: str
    Broker: str
    Asset: str
    
class Quote(BaseModel):
    stamp: datetime  # UTC-Zeitstempel, z.B. "2024-06-11T12:34:56Z"
    bid: float
    ask: float
    
class Position(BaseModel):
    asset: str
    position: int  # Anzahl der Anteile (positiv = Long, negativ = Short)
    open_price: float  # Eröffnungspreis der Position
    close_price: float  # Schlusskurs der Position
