from functools import lru_cache
from keras.models import load_model, Model
from pathlib import Path

@lru_cache(maxsize=1)
def latest() -> Model:
    path = Path('../model/latest.h5')
    return load_model('../model/latest.h5', compile=False) if path.exists() else None

@lru_cache(maxsize=1)
def best() -> Model:
    path = Path('../model/best.h5')
    return load_model('../model/best.h5', compile=False) if path.exists() else None