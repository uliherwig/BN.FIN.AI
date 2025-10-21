from pydantic import BaseModel

class AtrModel(BaseModel):
    period: int = 14  # Anzahl der Perioden für ATR-Berechnung
    threshold: float = 1.5  # Schwellenwert für ATR-Signale