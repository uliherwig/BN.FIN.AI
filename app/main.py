from fastapi import FastAPI
from celery import Celery
import threading
from app.api.v1 import alpaca_controller

app = FastAPI(
    title="FinAIService",
    version="1.0.0",
    docs_url="/swagger"
)

# API-Routen registrieren
app.include_router(alpaca_controller.router, prefix="/api/v1")

# Celery im selben Prozess (nur für Entwicklung!)
celery = Celery('tasks', broker='redis://redis:6379/0')
celery.conf.task_always_eager = False  # Tasks asynchron ausführen


