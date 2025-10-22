# ====================
# 몬테카를로 트리 탐색 생성
# ====================

# 패키지 임포트
from game import State
from dual_network import DN_INPUT_SHAPE
from math import sqrt
from keras.models import load_model
from pathlib import Path
import numpy as np
import time
import os

ADD_NOISE = os.getenv("ENV") == "SELF_PLAY"
print("ADD_NOISE: ", ADD_NOISE)


BATCH_SIZE = 16
_pending_states = []

DIR_NOISE_EPS = 0.25 # 엡실론
DIR_NOISE_ALPHA = 0.28 # 알파

def _flush_batch(model):
    global _pending_states
    if not _pending_states:
        return

    xs = []
    callbacks = []
    for s, cb in _pending_states:
        a, b, c = DN_INPUT_SHAPE
        x = np.array([s.pieces, s.enemy_pieces]).reshape(c, a, b).transpose(1, 2, 0)
        xs.append(x)
        callbacks.append(cb)
    xs = np.stack(xs, axis=0)


    y_p, y_v = model.predict(xs, batch_size=len(xs), verbose=0)

    for i, cb in enumerate(callbacks):
        cb(y_p[i], y_v[i][0])
    
    _pending_states = []

# 추론
def predict(model, state, on_ready):
    global _pending_states
    _pending_states.append((state, on_ready))

    if len(_pending_states) >= BATCH_SIZE:
        _flush_batch(model)



# 노드 리스트를 시행 횟수 리스트로 변환
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores


# 몬테카를로 트리 탐색 스코어 얻기
def pv_mcts_scores(model, state, temperature, evaluate_count):
    # 몬테카를로 트리 탐색 노드 정의
    class Node:
        # 노드 초기화
        def __init__(self, state, p, parent=None):
            self.state = state  # 상태
            self.p = p  # 정책
            self.w = 0  # 가치 누계
            self.n = 0  # 시행 횟수
            self.parent = parent
            self.child_nodes = None  # 子ノード群

        def expand(self, policies):
            if ADD_NOISE and self.parent is None:
                noise = np.random.dirichlet(
                    [DIR_NOISE_ALPHA] * len(policies)
                )
                policies = (1.0 - DIR_NOISE_EPS) * policies + DIR_NOISE_EPS * noise

            self.child_nodes = []
            for action, policy in zip(self.state.legal_actions(), policies):
                next_state = self.state.next(action)
                self.child_nodes.append(Node(next_state, policy, parent=self))
                

        # 국면 가치 누계
        def evaluate(self):
            # 게임 종료 시
            if self.state.is_done():
                # 승패 결과로 가치 얻기
                value = -1 if self.state.is_lose() else 0
                # 누계 가치와 시행 횟수 갱신
                self.w += value
                self.n += 1
                return value

            # 자녀 노드가 존재하지 않는 경우
            if not self.child_nodes:
                # 뉴럴 네트워크 추론을 활용한 정책과 가치 얻기
                policies = None
                value = None
                done = False

                def _cb(policies_raw, value_raw):
                    nonlocal done, policies, value

                    legal = list(self.state.legal_actions())
                    policies = policies_raw[legal]
                    policies /= sum(policies) if sum(policies) else 1
                    value = value_raw
                    done = True
                
                predict(model, self.state, _cb)

                while not done:
                    _flush_batch(model)
                    time.sleep(0.001)


                # 누계 가치와 시행 횟수 갱신
                self.w += value
                self.n += 1
                self.expand(policies)
                return value

            # 자녀 노드가 존재하지 않는 경우
            else:
                # 아크 평갓값이 가장 큰 자녀 노드를 평가해 가치 얻기
                value = -self.next_child_node().evaluate()

                # 누계 가치와 시행 횟수 갱신
                self.w += value
                self.n += 1
                return value

        # 아크 평가가 가장 큰 자녀 노드 얻기
        def next_child_node(self):
            # 아크 평가 계산
            C_PUCT = 1.0
            t = sum(nodes_to_scores(self.child_nodes))
            pucb_values = []
            for child_node in self.child_nodes:
                pucb_values.append((-child_node.w / child_node.n if child_node.n else 0.0) +
                                   C_PUCT * child_node.p * sqrt(t) / (1 + child_node.n))

            # 아크 평갓값이 가장 큰 자녀 노드 반환
            return self.child_nodes[np.argmax(pucb_values)]


    # 현재 국면의 노드 생성
    root_node = Node(state, 0)

    # 여러 차례 평가 실행
    for _ in range(evaluate_count):
        root_node.evaluate()
    
    _flush_batch(model)

    # 합법적인 수의 확률 분포
    scores = nodes_to_scores(root_node.child_nodes)
    if temperature == 0:  # 최대값인 경우에만 1
        action = np.argmax(scores)
        scores = np.zeros(len(scores))
        scores[action] = 1
    else:  # 볼츠만 분포를 기반으로 분산 추가
        scores = boltzman(scores, temperature)
    return scores


# 몬테카를로 트리 탐색을 활용한 행동 선택
def pv_mcts_action(model, temperature=0, evaluate_count=30):
    def pv_mcts_action(state):
        scores = pv_mcts_scores(model, state, temperature, evaluate_count)
        return np.random.choice(state.legal_actions(), p=scores)

    return pv_mcts_action


# 볼츠만 분포
def boltzman(xs, temperature):
    xs = [x ** (1 / temperature) for x in xs]
    return [x / sum(xs) for x in xs]


# 동작 확인
if __name__ == '__main__':
    # 모델 로드
    path = sorted(Path('../model').glob('*.h5'))[-1]
    model = load_model(str(path))

    # 상태 생성
    state = State()

    # 몬테카를로 트리 탐색을 활용해 행동을 얻는 함수 생성
    next_action = pv_mcts_action(model, 1.0)

    # 게임 종료 시까지 반복
    while True:
        # 게임 종료 시
        if state.is_done():
            break

        # 행동 얻기
        action = next_action(state)

        # 다음 상태 얻기
        state = state.next(action)

        # 문자열 출력
        print(state)
