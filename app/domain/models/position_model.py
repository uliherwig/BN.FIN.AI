
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain.models.enums import SideEnum, IndicatorEnum


class PositionModel(BaseModel):
    id: UUID
    strategy_id: UUID
    strategy_type: IndicatorEnum
    execution_id: UUID 
    symbol: str
    quantity: Decimal
    side: SideEnum
    price_open: Decimal
    price_close: Decimal
    profit_loss: Decimal
    take_profit: Decimal
    stop_loss: Decimal
    stamp_opened: datetime
    stamp_closed: datetime
    close_signal: str
    strategy_params: str
    confidence_up: Decimal = Decimal(0)
    confidence_down: Decimal = Decimal(0)
