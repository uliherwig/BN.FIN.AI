from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime, timezone
from app.domain.models.enums import SideEnum, IndicatorEnum

class StrategyResultModel(BaseModel):
    strategy_id: UUID   
    strategy_type: IndicatorEnum
    asset: str    
    quantity: Decimal = Decimal('1.0')
    take_profit_pct: float = 0.0
    stop_loss_pct: float = 0.0
    strategy_params: str = "{}"  # JSON string for strategy-specific parameters
    number_of_positions: int = 0
    profit: Decimal = Decimal('0.0')
    number_of_short_positions: int = 0
    number_of_long_positions: int = 0
    number_of_positive_positions: int = 0
    number_of_negative_positions: int = 0