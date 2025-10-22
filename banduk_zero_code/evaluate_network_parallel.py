import ray
from game import State
from pv_mcts import pv_mcts_action
from shutil import copy
from model_pool import latest, best
from dual_network import build_model
import config


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


EN_GAME_COUNT = 50
EN_TEMPERATURE = 0
NUM_ACTORS = 12


def first_player_point(ended_state:State):
    # 1: 선 수 플레이어 승리, 0: 선 수 플레이어 패배
    if ended_state.is_first_player():
        return 0 if ended_state.is_lose() else 1
    return 1 if ended_state.is_lose() else 0

@ray.remote(num_cpus=1, num_gpus=0)
class EvaluatorActor:
    def __init__(self, latest_weights_ref, best_weights_ref):
        
        import tensorflow as tf
        print("Available physical devices:", tf.config.list_physical_devices('GPU'))
        latest_model, best_model = build_model(), build_model()

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

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=12)
    
    # evaluate_count = config.SP_EVALUATE_COUNT
    evaluate_count = 100


    best.cache_clear(); latest.cache_clear()
    latest_model_ref = ray.put(latest().get_weights())
    best_model_ref = ray.put(best().get_weights())
    
    actors = [EvaluatorActor.remote(latest_model_ref, best_model_ref) for _ in range(NUM_ACTORS)]



    futures = []
    for i in range(EN_GAME_COUNT):
        actor = actors[i % NUM_ACTORS]
        futures.append(actor.play_game.remote(i, evaluate_count))



    print(f"\rEvaluate 0/{EN_GAME_COUNT}", end="", flush=True)
    results = []
    finished = 0

    while futures:
        done, futures = ray.wait(futures, num_returns=1)
        results.append(ray.get(done[0]))
        finished = EN_GAME_COUNT - len(futures)
        print(f"\rEvaluate {finished}/{EN_GAME_COUNT}", end="", flush=True)
    print()

    avg = sum(results) / EN_GAME_COUNT
    print(f"AveragePoint: {avg:.3f}")

    if avg >= 0.6:
        copy('../model/latest.h5', '../model/best.h5')
        best.cache_clear()
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