from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, StrategyLibEnum

class TemaModel(BaseModel):
    short_ma: int
    long_ma: int
