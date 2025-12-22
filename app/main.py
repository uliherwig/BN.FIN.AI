from fastapi import FastAPI
from app.interfaces.api.v1 import alpaca_controller

app = FastAPI(
    title="FinAIService",
    version="1.0.0",
    docs_url="/swagger"
)

# register routers
app.include_router(alpaca_controller.router, prefix="/api/v1")

# include a redis subscription as an example
from app.infrastructure.redis_service import get_redis_service  
redis_service = get_redis_service()
def message_handler(message):
    print(f"Received message: {message['data']}")

redis_service.subscribe("my_channel", message_handler)






