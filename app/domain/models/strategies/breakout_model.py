from pydantic import BaseModel

class BreakoutModel(BaseModel):
    breakout_period: int = 20  # Anzahl der Perioden für High/Low Berechnung
