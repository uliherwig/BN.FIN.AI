from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from app.domain.models.enums import SideEnum, StrategyLibEnum, StrategyLibEnum

class StrategySettingsModel(BaseModel):
    id: UUID   
    strategy_type: StrategyLibEnum
    start_date: datetime = Field(default_factory=lambda: datetime(1, 1, 1, 0, 0, 0, tzinfo=timezone.utc), alias="StartDate")
    end_date: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), alias="EndDate")
    asset: str = Field(alias="Asset")
    quantity: Decimal = Decimal('1.0')
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.01
    strategy_params: str = "{}"  # JSON string for strategy-specific parameters
    close_positions_eod: bool = True
    
    class Config:
        populate_by_name = True
        
        
        
        

    # public Guid Id { get; set; } = Guid.NewGuid();
    # public DateTime StampStart { get; set; } = DateTime.UtcNow.ToUniversalTime();
    # public DateTime StampEnd { get; set; } = new DateTime(1, 1, 1, 0, 0, 0).ToUniversalTime();
    # public Guid UserId { get; set; }
    # public StrategyEnum StrategyType { get; set; }
    # public string Broker { get; set; }
    # public string Name { get; set; }
    # public string Asset { get; set; }
    # public int Quantity { get; set; }
    # public decimal TakeProfitPercent { get; set; }
    # public decimal StopLossPercent { get; set; }
    # public DateTime StartDate { get; set; }
    # public DateTime EndDate { get; set; }
    # public decimal TrailingStop { get; set; } = 0m;
    # public bool ClosePositionEod { get; set; } = true;
    # public bool Bookmarked { get; set; } = false;
    # public bool Optimized { get; set; } = false;
    # public string StrategyParams { get; set; } = string.Empty;
