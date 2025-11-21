from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, IndicatorEnum

class RsiModel(BaseModel):
    RSI_period: int = 14      # Standard-RSI-Periode
    RSI_overbought: int = 70  # Überkauft-Schwelle (Standard: 70)
    RSI_oversold: int = 30    # Überverkauft-Schwelle (Standard: 30)
