from decimal import ROUND_HALF_UP, Decimal
from uuid import UUID
import uuid
from app.domain.models.enums import SideEnum, StrategyLibEnum
from app.domain.models.position_model import PositionModel
from datetime import datetime, timezone, time



class PositionManager:
    def __init__(self):       
        self.positions: dict[UUID,PositionModel] = {} 
        self.strategy_id: UUID = UUID(int=0)
        self.asset: str = ""
        self.strategy_type: StrategyLibEnum = StrategyLibEnum.NONE
        self.strategy_params: str = ""
        self.quantity: Decimal = Decimal(0)   
        self.close_positions_eod: bool = True

    @classmethod
    def create_with_test_params(cls, asset: str, quantity: Decimal, strategy_type: StrategyLibEnum, strategy_params: str = "", close_positions_eod: bool = True):
        """Alternativer Constructor mit Strategy"""
        instance = cls()
        instance.positions = {}
        instance.asset = asset
        instance.strategy_type = strategy_type
        instance.strategy_params = strategy_params
        instance.quantity = quantity
        instance.close_positions_eod = close_positions_eod
        return instance


    def open_position(self,
                      side: SideEnum,
                      price: Decimal,
                      tp:Decimal,
                      sl:Decimal,
                      stamp : datetime
                      ):
        position_id = uuid.uuid4()
        position = PositionModel(
            id=position_id,
            strategy_id=self.strategy_id,
            strategy_type=self.strategy_type,
            execution_id=UUID(int=0),
            symbol=self.asset,
            quantity=self.quantity,
            side=side,
            price_open=price,
            price_close=Decimal(0),
            profit_loss=Decimal(0),
            take_profit=tp,
            stop_loss=sl,
            stamp_opened= stamp,
            stamp_closed=datetime(1970, 1, 1, tzinfo=timezone.utc),
            close_signal="",
            strategy_params=self.strategy_params
        )
        self.positions[position_id] = position
        return position_id

    def get_positions(self):
        return self.positions
    
    def get_position(self, position_id: UUID):
        return self.positions.get(position_id, None)

    def update_position(self, position_id: int, status: str):
        # if 0 <= position_id < len(self.positions):
        #     self.positions[position_id]["status"] = status
        #     return self.positions[position_id]
        return None

    def execute_sl_tp(self, position_id: UUID, bid: Decimal, ask: Decimal, stamp: datetime):
        position = self.get_position(position_id)  
        
        if position and position.side == SideEnum.Buy:
            
            # Check for TakeProfit
            if bid >  position.take_profit:
                self.close_position(position_id, bid, stamp)
                return UUID(int=0)
            
            # Check for StopLoss
            if bid < position.stop_loss:
                self.close_position(position_id, bid, stamp)
                return UUID(int=0)

        if position and position.side == SideEnum.Sell:   
            
            # Check for TakeProfit
            if ask < position.take_profit:
                self.close_position(position_id, ask, stamp)
                return UUID(int=0)

            # Check for StopLoss
            if ask > position.stop_loss:
                self.close_position(position_id, ask,stamp)
                return UUID(int=0)

        return position_id

    def close_position(self, position_id: UUID, price: Decimal, stamp: datetime):
        position = self.get_position(position_id)
        if position:
            position.stamp_closed = stamp
            position.price_close = price
            price_close = Decimal(str(position.price_close))
            price_open = Decimal(str(position.price_open))
            quantity = Decimal(str(position.quantity))
            side_multiplier = 1 if position.side == SideEnum.Buy else -1
            profit_loss = (price_close - price_open) * quantity * side_multiplier
       
            position.profit_loss = profit_loss.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        return UUID(int=0)
    
    def calculate_profit(self):
        return sum(
            (position.profit_loss for position in self.positions.values()),
            start=Decimal(0)
        )