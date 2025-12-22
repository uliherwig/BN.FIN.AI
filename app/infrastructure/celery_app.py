from celery import Celery

REDIS_URL = GLOBAL_CONFIG["REDIS_URL"]

def create_celery_app():
    celery_app = Celery('strategy_tasks')
    celery_app.config_from_object({
        'broker_url': REDIS_URL,
        'result_backend': REDIS_URL,
        'task_serializer': 'json',
        'accept_content': ['json'],
        'result_serializer': 'json',
        'timezone': 'UTC',
        'enable_utc': True,
    })
    return celery_app


celery_app = create_celery_app()
