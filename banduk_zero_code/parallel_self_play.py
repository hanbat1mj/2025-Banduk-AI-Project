import ray
import pickle
import os
from datetime import datetime
from self_play_worker import SelfPlayActor
from model_pool import best
import config

finished_games = 0
NUM_ACTORS = 12

def write_data(index, history):
    global finished_games

    now = datetime.now()
    os.makedirs('../data/', exist_ok=True)
    path = f"../data/{now.strftime('%Y%m%d%H%M%S')}_game{index}.history"
    with open(path, 'wb') as f:
        pickle.dump(history, f)
    print(f"\rSelf-Play {finished_games}/{config.SP_GAME_COUNT}", end="", flush=True)

def run_self_play():
    if (config.SP_GAME_COUNT <= 0):
        config.set_game_count(64)
        return
    global finished_games

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True, num_cpus=12)

    best.cache_clear()
    best_weight_ref = ray.put(best().get_weights())
    
    actors = [SelfPlayActor.remote(best_weight_ref) for _ in range(NUM_ACTORS)]
    futures = [
        actors[i % NUM_ACTORS].play_game.remote(i, config.SP_EVALUATE_COUNT)
        for i in range(config.SP_GAME_COUNT)
    ]
    
    finished_games = 0 # 전체 완료 게임 수
    batch_counter = 0 # 저장한 배치 번호
    buffer = [] # history 누적 버퍼

    print(f"\rSelf-Play 0/{config.SP_GAME_COUNT}", end="", flush=True)

    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        history_single = ray.get(done[0])

        buffer.extend(history_single)
        finished_games += 1

        print(f"\rSelf-Play {finished_games}/{config.SP_GAME_COUNT}", end="", flush=True)

        if finished_games % config.SAVE_INTERVAL == 0:
            batch_counter += 1
            write_data(batch_counter, buffer)
            buffer = []
    
    if buffer:
        batch_counter += 1
        write_data(batch_counter, buffer)
    print()

if __name__ == '__main__':
    run_self_play()
    ray.shutdown()