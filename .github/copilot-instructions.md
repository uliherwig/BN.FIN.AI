# Copilot Instructions for BN.FIN.AI

## Project Overview
- **BN.FIN.AI** is a Python-based platform for testing and optimizing trading strategies, combining traditional and AI-driven approaches.
- The system is modular, with clear separation between API, domain logic, infrastructure, and model definitions.
- **FastAPI** is used for the web API (`app/main.py`), with endpoints defined in `app/api/v1/alpaca_controller.py`.
- **Celery** (with Redis) is used for asynchronous and background task execution, especially for strategy testing and optimization.

## Key Components
- **Domain Logic**: Core trading logic, strategy management, and AI/ML models are in `app/domain/`.
  - `ai_trader.py`: Implements AI-based trading logic and feature engineering.
  - `strategy_manager.py`: Handles strategy backtesting and feature selection.
  - `services/`: Contains submodules for training, optimization, and testing services.
- **API Layer**: REST endpoints for strategy operations are in `app/api/v1/alpaca_controller.py`.
- **Infrastructure**: Integration with external services (Alpaca, Redis, Yahoo) is in `app/infrastructure/`.
- **Models**: Data schemas and enums are in `app/domain/models/` and `app/models/schemas.py`.

## Developer Workflows
- **Run the API**: `uvicorn app.main:app --host 0.0.0.0 --port 8000` (see `Dockerfile` for containerized setup).
- **Celery Worker**: Start with `celery -A app.domain.services.strategy_service.celery_app worker --loglevel=info` (adjust path as needed).
- **Testing Strategies**: Use the `/api/v1/test-strategy` endpoint (see example JSON in `alpaca_controller.py`).
- **Optimize Strategies**: Use `/api/v1/optimize-strategy` endpoint; optimization history is saved as HTML.

## Project Conventions & Patterns
- **Data Flow**: Market data is loaded from Redis or CSV, processed by domain services, and results are saved to CSV for analysis.
- **Strategy Settings**: Passed as Pydantic models or dicts, often via API endpoints.
- **Feature Engineering**: Technical indicators are modularized in `operations/indicators/` and selected dynamically.
- **Walk-Forward Validation**: Used for model evaluation (see `ai_trader.py`, `strategy_manager.py`).
- **Async Tasks**: Prefer Celery for long-running or resource-intensive jobs.
- **File Naming**: CSVs and results are named with asset, date, and interval for traceability.

## Integration Points
- **Alpaca**: Used for market data and trading simulation (see `infrastructure/alpaca_service.py`).
- **Redis**: Used for caching and as a Celery broker.
- **Optuna**: For hyperparameter optimization (see `services/optimize/`).
- **LightGBM, TensorFlow, scikit-learn**: Used for ML models.

## Examples
- To add a new indicator: Implement in `operations/indicators/`, register in `indicator_factory.py`.
- To add a new strategy: Extend `StrategyManager` and update API/controller as needed.

## References
- See `README.md` for a high-level summary.
- See `app/api/v1/alpaca_controller.py` for API usage examples and expected JSON payloads.
- See `app/domain/services/strategy_service.py` for Celery task definitions and optimization logic.

---
For questions or unclear conventions, review the referenced files or ask for clarification.
