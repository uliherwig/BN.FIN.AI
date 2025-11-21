from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, IndicatorEnum

class RocModel(BaseModel):

    ROC_period: int
    ROC_threshold: Decimal =  Decimal('0.001')

