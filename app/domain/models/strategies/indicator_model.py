from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, IndicatorEnum


class IndicatorModel(BaseModel):
    strategyType: IndicatorEnum
    params: str
