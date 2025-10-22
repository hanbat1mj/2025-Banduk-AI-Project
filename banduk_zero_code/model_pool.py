# 모델 캐싱을 위한 풀(Pool) 모듈
# 학습된 모델을 메모리에 효율적으로 로드하고 재사용하기 위한 기능 제공

from functools import lru_cache
from keras.models import load_model, Model
from pathlib import Path

@lru_cache(maxsize=1)  # 최근 로드된 모델을 캐시에 저장하여 중복 로딩 방지
def latest() -> Model:
    """가장 최근에 학습된 모델을 반환
    Returns:
        Model: 최신 모델 또는 파일이 없는 경우 None
    """
    path = Path('../model/latest.h5')
    return load_model('../model/latest.h5', compile=False) if path.exists() else None

@lru_cache(maxsize=1)  # 최근 로드된 모델을 캐시에 저장하여 중복 로딩 방지
def best() -> Model:
    """현재까지 가장 성능이 좋은 모델을 반환
    Returns:
        Model: 최고 성능 모델 또는 파일이 없는 경우 None
    """
    path = Path('../model/best.h5')
    return load_model('../model/best.h5', compile=False) if path.exists() else None