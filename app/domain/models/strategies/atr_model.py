from pydantic import BaseModel



class AtrModel(BaseModel):

    ATR_period: int = 14  # Anzahl der Perioden für ATR-Berechnung
    ATR_threshold: float = 1.5  # Schwellenwert für ATR-Signale # Anzahl der Perioden für ATR-Berechnung
