from pydantic import BaseModel

class SmaModel(BaseModel):
    short_ma: int
    long_ma: int
