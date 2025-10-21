from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, StrategyLibEnum

class RsiModel(BaseModel):
    period: int = 14      # Standard-RSI-Periode
    overbought: int = 70  # Überkauft-Schwelle (Standard: 70)
    oversold: int = 30    # Überverkauft-Schwelle (Standard: 30)
