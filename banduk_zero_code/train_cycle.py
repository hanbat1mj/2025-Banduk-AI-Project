# ====================
# 학습 사이클 실행 (Ray 병렬 self-play 포함)
# ====================

# 패키지 임포트
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_GRAPH_REWRITE"] = "0"
os.environ["ENV"] = "SELF_PLAY"

from dual_network import dual_network
from train_network import train_network
from evaluate_network_parallel import evaluate_network_parallel
from parallel_self_play import run_self_play
from config import set_evaluate_count, set_game_count, set_save_interval, SAVE_INTERVAL

from datetime import datetime
import time

import ray
import argparse
import subprocess

PASS_SP_COUNT = 0
PASS_TRAIN = 0

def print_ended_time(what_schedule: str, elapsed=None):

    print(f"\n=== {what_schedule} Ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    if elapsed != None:
        print(f"=== Duration: {int(elapsed // 60)} min {int(elapsed % 60)} sec ===")



# 듀얼 네트워크 생성 (최초 1회만 실행됨)
def train_cycle():

    global PASS_SP_COUNT
    global PASS_TRAIN


    dual_network()

    # ray 생성
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=12)

    # 학습 사이클 실행
    for i in range(10):
        
        timer_start = time.time()
        elapsed = 0.0

        print('Train', i+1, '====================')
        print("\n=== Train", i+1, f"Cycle Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        
        '''
        한 사이클 당 파라미터를 실험적으로 조절하여 Train 가능.
        if i % 2 == 0:
            set_evaluate_count(100)
            set_game_count(48)
        else:
            set_evaluate_count(200)
            set_game_count(24)
        '''

        set_save_interval(6)
        set_evaluate_count(80)

        set_game_count(64 if PASS_SP_COUNT == 0 else 64-PASS_SP_COUNT*SAVE_INTERVAL)

        run_self_play()
        if PASS_SP_COUNT == 0:
            elapsed = time.time() - timer_start
            print_ended_time("Self Play", elapsed)
            timer_start = time.time()
        PASS_SP_COUNT = 0

        # 파라미터 학습
        if (PASS_TRAIN == 0):
            subprocess.run(["python", "train_network.py"])
            elapsed = time.time() - timer_start
            print_ended_time("Train Network", elapsed)
            timer_start = time.time()
        else:
            PASS_TRAIN = 0

        # 베스트 모델과 비교 평가
        subprocess.run(["python", "evaluate_network_parallel.py"], env={**os.environ, "ENV": "EVAL"})
        elapsed = time.time() - timer_start
        print_ended_time("Evaluate Network", elapsed)
        timer_start = time.time()

    ray.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass_sp", type=int, default=0, help="Self-play 판 수")
    parser.add_argument("--pass_tr", type=int, default=0, help="train 스킵할지:1 안할지:0")


    args = parser.parse_args()
    PASS_SP_COUNT = args.pass_sp
    PASS_TRAIN = args.pass_tr


    train_cycle()

