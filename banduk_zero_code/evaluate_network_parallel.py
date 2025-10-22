# ====================
# 네트워크 평가 병렬 처리 모듈
# Ray를 사용하여 여러 게임을 동시에 실행하고 새 모델과 최고 모델을 비교 평가
# ====================

import ray  # 분산 컴퓨팅 프레임워크
from game import State
from pv_mcts import pv_mcts_action
from shutil import copy
from model_pool import latest, best
from dual_network import build_model
import config

# 텐서플로우 멀티스레딩 설정
# Ray worker들이 CPU 리소스를 효율적으로 사용하도록 각 프로세스의 스레드 수 제한
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 평가 관련 상수
EN_GAME_COUNT = 50     # 평가전 진행 횟수
EN_TEMPERATURE = 0     # 탐색 온도 (0: 최선의 수만 선택)
NUM_ACTORS = 12        # 동시 실행할 평가자 수


def first_player_point(ended_state:State):
    # 1: 선 수 플레이어 승리, 0: 선 수 플레이어 패배
    if ended_state.is_first_player():
        return 0 if ended_state.is_lose() else 1
    return 1 if ended_state.is_lose() else 0

# Ray Actor 클래스 정의
# 각 액터는 독립적인 프로세스에서 실행되며 자체 모델 복사본을 가짐
@ray.remote(num_cpus=1, num_gpus=0)  # 각 액터는 1개의 CPU 코어 사용
class EvaluatorActor:
    def __init__(self, latest_weights_ref, best_weights_ref):
        """액터 초기화
        Args:
            latest_weights_ref: Ray 객체 저장소에 있는 최신 모델 가중치
            best_weights_ref: Ray 객체 저장소에 있는 최고 모델 가중치
        """
        import tensorflow as tf
        print("Available physical devices:", tf.config.list_physical_devices('GPU'))
        
        # 각 액터별로 독립적인 모델 인스턴스 생성
        latest_model, best_model = build_model(), build_model()

        # Ray 객체 저장소에서 가중치를 가져와 모델에 설정
        latest_model.set_weights(latest_weights_ref)
        best_model.set_weights(best_weights_ref)
        
        self.latest_model = latest_model
        self.best_model = best_model
    
    def play_game(self, idx: int, evaluate_count: int) -> int:
        first_model, second_model = (
            (self.latest_model, self.best_model)
            if idx % 2 == 0 else
            (self.best_model, self.latest_model)
        )


        is_latest_first = (idx % 2 == 0)

        m0_act = pv_mcts_action(first_model, EN_TEMPERATURE, evaluate_count)
        m1_act = pv_mcts_action(second_model, EN_TEMPERATURE, evaluate_count)

        s = State()


        
        while not s.is_done():
            
            depth = s.depth
            def adaptive_evaluate_count(depth: int) -> int:
                if depth <= 6:
                    return 400  # 초반 깊이 0~4 → legal moves ~30~36 → 10*30=300
                elif depth <= 16:
                    return 200  # 중반 5~14
                elif depth <= 26:
                    return 150  # 중후반 15~24
                else:
                    return  100  # 종반 25~35
            sp_evaluate_count = adaptive_evaluate_count(depth)

            m0_act = pv_mcts_action(first_model, EN_TEMPERATURE, sp_evaluate_count)
            m1_act = pv_mcts_action(second_model, EN_TEMPERATURE, sp_evaluate_count)

            s = s.next(m0_act(s) if s.is_first_player() else m1_act(s))
        
        if is_latest_first:
            print("latest seon ", "win" if first_player_point(s) == 1 else "lose")
            return first_player_point(s)
        else:
            print("best seon ", "win" if first_player_point(s) != 1 else "lose")
            return 1 - first_player_point(s)


def evaluate_network_parallel() -> bool:
    """병렬로 네트워크 평가를 실행하는 메인 함수
    
    Returns:
        bool: 최신 모델이 최고 모델을 이겼을 경우 True
    """
    # Ray 초기화 (12개 CPU 코어 사용)
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=12)
    
    # MCTS 탐색 횟수 설정
    evaluate_count = 100

    # 모델 캐시 초기화 및 가중치 로드
    best.cache_clear(); latest.cache_clear()
    # 모델 가중치를 Ray 객체 저장소에 저장 (모든 액터가 공유)
    latest_model_ref = ray.put(latest().get_weights())
    best_model_ref = ray.put(best().get_weights())
    
    # 여러 개의 평가자 액터 생성
    actors = [EvaluatorActor.remote(latest_model_ref, best_model_ref) for _ in range(NUM_ACTORS)]

    # 비동기로 게임 실행 작업 제출
    futures = []
    for i in range(EN_GAME_COUNT):
        actor = actors[i % NUM_ACTORS]  # 라운드 로빈 방식으로 액터 선택
        futures.append(actor.play_game.remote(i, evaluate_count))  # 비동기 실행



    # 진행상황 표시 초기화
    print(f"\rEvaluate 0/{EN_GAME_COUNT}", end="", flush=True)
    results = []
    finished = 0

    # 비동기로 실행된 게임들의 결과 수집
    while futures:
        # 완료된 게임 결과 대기 (1개씩 처리)
        done, futures = ray.wait(futures, num_returns=1)
        results.append(ray.get(done[0]))  # 완료된 게임의 결과 수집
        finished = EN_GAME_COUNT - len(futures)
        print(f"\rEvaluate {finished}/{EN_GAME_COUNT}", end="", flush=True)
    print()

    # 승률 계산 및 모델 교체 여부 결정
    avg = sum(results) / EN_GAME_COUNT
    print(f"AveragePoint: {avg:.3f}")

    # 승률이 60% 이상이면 최신 모델을 최고 모델로 교체
    if avg >= 0.6:
        copy('../model/latest.h5', '../model/best.h5')
        best.cache_clear()  # 캐시된 최고 모델 초기화
        print('Change BestPlayer')
        changed = True
    else:
        print('Keep BestPlayer')
        changed = False


    return changed


if __name__ == '__main__':
    try:
        evaluate_network_parallel()
    finally:
        ray.shutdown()