import optuna
from optuna.storages.journal import JournalStorage, JournalRedisBackend

REDIS_URL = "redis://localhost:6379/0"


def create_redis_storage():
    backend = JournalRedisBackend(REDIS_URL)
    return JournalStorage(backend)


def create_study(study_name: str, directions):
    """
    Helper to create / load Optuna study via Redis.
    """
    storage = create_redis_storage()
    
    optuna.delete_study(storage=storage, study_name=study_name)

    return optuna.create_study(
        study_name=study_name,
        directions=directions,
        storage=storage,
        load_if_exists=False
    )
