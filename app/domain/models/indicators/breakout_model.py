from pydantic import BaseModel

class BreakoutModel(BaseModel):
    Breakout_period: int = 20  # Anzahl der Perioden für High/Low Berechnung
