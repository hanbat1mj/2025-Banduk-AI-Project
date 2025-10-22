# ====================
# 파라미터 갱신 파트
# ====================

from dual_network import DN_INPUT_SHAPE
from keras.callbacks import LearningRateScheduler, LambdaCallback, TensorBoard
from keras import backend as K
from pathlib import Path
import numpy as np
import pickle
from model_pool import latest, best
import config
from math import ceil
import datetime

import tensorflow as tf

# tf.config.threading.set_intra_op_parallelism_threads(32)
# tf.config.threading.set_inter_op_parallelism_threads(4)

# 파라미터 준비
RN_EPOCHS = 3  # 학습 횟수


# 학습 데이터 로드
def load_data():

    # needed_files = ceil(config.SP_GAME_COUNT / config.SAVE_INTERVAL)
    needed_files = ceil(config.SP_GAME_COUNT / config.SAVE_INTERVAL) * 2
    history_paths = sorted(Path('../data').glob('*.history'))[-needed_files:]
    print(needed_files)
    print(history_paths)
    history = []

    for path in history_paths:
        with path.open(mode='rb') as f:
            history.extend(pickle.load(f))
    print("History Length: ", len(history))
    return history


# 듀얼 네트워크 학습
def train_network():

    # tf.config.threading.set_intra_op_parallelism_threads(6)
    # tf.config.threading.set_inter_op_parallelism_threads(2)

    print(tf.config.threading.get_intra_op_parallelism_threads())
    print(tf.config.threading.get_inter_op_parallelism_threads())

    # 학습 데이터 로드
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    # 학습을 위한 입력 데이터 셰이프로 변환
    a, b, c = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b).transpose(0, 2, 3, 1)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values)

    # 베스트 플레이어 모델 로드
    best.cache_clear(); latest.cache_clear()
    model = best()

    # 모델 컴파일
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='adam',
                  loss_weights=[1.0, 0.25])

    # 학습률
    def step_decay(epoch):
        progress = epoch / RN_EPOCHS
        if progress < 0.33:
            return 0.001
        elif progress < 0.66:
            return 0.0005
        else:
            return 0.00025

    lr_decay = LearningRateScheduler(step_decay)

    print_loss = LambdaCallback(
        on_epoch_end=lambda epoch, logs:
        print(
            f"\n"
            f"Epoch {epoch+1}/{RN_EPOCHS} "
            f"total_loss={logs['loss']:.4f} "
            f"policy_loss={logs.get('pi_loss', logs.get('categorical_crossentropy_loss', logs.get('policy_output_loss'))) : .4f} "
            f"value_loss={logs.get('v_loss', logs.get('mse_loss', logs.get('value_output_loss'))) : .4f}"

        )
    )

    # 출력
    print_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs:
        print('\rTrain {}/{}'.format(epoch + 1, RN_EPOCHS), end=''))
    
    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # 학습 실행
    model.fit(xs, [y_policies, y_values], batch_size=64, shuffle=True, epochs=RN_EPOCHS,
              verbose=0, callbacks=[lr_decay, print_callback, print_loss, tensorboard_callback])
    print('')

    # 최신 플레이어 모델 저장
    model.save('../model/latest.h5')
    latest.cache_clear()
    

    # 모델 파기
    K.clear_session()
    del model


# 동작 확인
if __name__ == '__main__':
    train_network()