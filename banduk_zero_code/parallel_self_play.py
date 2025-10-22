# ====================
# 병렬 셀프 플레이 실행 모듈
# Ray를 사용하여 여러 개의 게임을 동시에 진행하고 학습 데이터를 생성
# ====================

import ray
import pickle
import os
from datetime import datetime
from self_play_worker import SelfPlayActor
from model_pool import best
import config

# 전역 변수 설정
finished_games = 0  # 완료된 게임 수 추적
NUM_ACTORS = 12    # 동시에 실행할 게임 프로세스 수

def write_data(index, history):
    """게임 히스토리를 파일로 저장하는 함수
    
    Args:
        index (int): 저장할 배치 번호
        history (list): 게임 진행 기록 데이터
    """
    global finished_games

    now = datetime.now()
    os.makedirs('../data/', exist_ok=True)
    path = f"../data/{now.strftime('%Y%m%d%H%M%S')}_game{index}.history"
    with open(path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\rSelf-Play {finished_games}/{config.SP_GAME_COUNT}", end="", flush=True)

def run_self_play():
    """병렬 셀프 플레이 실행 함수
    
    Ray를 사용하여 NUM_ACTORS 개수만큼의 게임을 동시에 실행하고
    각 게임의 결과를 수집하여 학습 데이터로 저장
    """
    if (config.SP_GAME_COUNT <= 0):
        config.set_game_count(64)
        return
    global finished_games

    # Ray 초기화 (분산 컴퓨팅 프레임워크)
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, num_cpus=12)

    # 최신 모델 가중치 로드 및 캐시 초기화
    best.cache_clear()
    best_weight_ref = ray.put(best().get_weights())
    
    # 병렬 처리를 위한 액터(프로세스) 생성
    actors = [SelfPlayActor.remote(best_weight_ref) for _ in range(NUM_ACTORS)]
    
    # 각 액터에게 게임 실행 작업 분배
    futures = [
        actors[i % NUM_ACTORS].play_game.remote(i, config.SP_EVALUATE_COUNT)
        for i in range(config.SP_GAME_COUNT)
    ]
    
    # 게임 진행 상태 추적 변수 초기화
    finished_games = 0  # 전체 완료 게임 수
    batch_counter = 0   # 저장한 배치 번호
    buffer = []         # history 누적 버퍼

    print(f"\rSelf-Play 0/{config.SP_GAME_COUNT}", end="", flush=True)

    # 모든 게임이 완료될 때까지 결과 수집
    while futures:
        # 완료된 게임 결과 대기
        done, futures = ray.wait(futures, num_returns=1)
        history_single = ray.get(done[0])

        # 게임 결과를 버퍼에 추가하고 진행상황 업데이트
        buffer.extend(history_single)
        finished_games += 1

        print(f"\rSelf-Play {finished_games}/{config.SP_GAME_COUNT}", end="", flush=True)

        # 설정된 간격마다 데이터 저장
        if finished_games % config.SAVE_INTERVAL == 0:
            batch_counter += 1
            write_data(batch_counter, buffer)
            buffer = []
    
    # 남은 데이터가 있으면 마지막으로 저장
    if buffer:
        batch_counter += 1
        write_data(batch_counter, buffer)
    print()

if __name__ == '__main__':
    run_self_play()
    ray.shutdown()  # Ray 리소스 정리