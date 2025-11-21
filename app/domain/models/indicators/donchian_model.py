from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel
from datetime import datetime
from app.domain import *

class DonchianModel(BaseModel):
    DONCHIAN_window: int
