from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, StrategyLibEnum

class MacdModel(BaseModel):

    fast: int
    slow: int
    signal: int
