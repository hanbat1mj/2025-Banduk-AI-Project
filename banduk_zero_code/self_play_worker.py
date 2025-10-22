from game import State
from pv_mcts import pv_mcts_scores
import numpy as np
import ray
from dual_network import DN_OUTPUT_SIZE, build_model

    
# 선 수 플레이어 가치
def first_player_value(ended_state):
    # 1: 선 수 플레이어 승리, -1: 선 수 플레이어 패배, 0: 무승부
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

@ray.remote(num_cpus=1)
class SelfPlayActor:
    def __init__(self, best_weights_ref):
        
        best_model = build_model()
        best_model.set_weights(best_weights_ref)
        
        self.model = best_model
        '''
        self.model = latest() if random.random() < 0.5 else best()
        if self.model is None:
            self.model = best()
        '''
    
    def play_game(self, game_id: int, sp_evaluate_count: int):
        state = State()
        history = []
        progress = 0

        while not state.is_done():
            progress += 1
            temperature = 0 if progress >= 14 else 1.0
            # temperature = 1
            depth = state.depth
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

            # [임시] 방편으로 일단 이렇게 처리.

            scores = pv_mcts_scores(self.model, state, temperature, sp_evaluate_count)
            policies = [0] * DN_OUTPUT_SIZE
            for action, policy in zip(state.legal_actions(), scores):
                policies[action] = policy

            legal_actions = state.legal_actions()
            action = np.random.choice(legal_actions, p=scores)

            policies = np.zeros(DN_OUTPUT_SIZE, dtype=np.float32)
            policies[legal_actions] = scores

            history.append([[state.pieces, state.enemy_pieces], policies, None])

            state = state.next(action)
        print(progress)

        # value = -1 if state.is_lose() else 1
        value = first_player_value(state)

        augmented = []
        for [s, e], p_full, _ in history:

            f = np.asarray([s, e], dtype=np.float32).reshape(2, 6, 6)

            board_prob = p_full[:-1].reshape(6, 6)
            pass_prob = p_full[-1]

            v = value
            value = -value

            for k in range(4):
                f_rot = np.rot90(f, k, axes=(1, 2))
                b_rot:np.ndarray = np.rot90(board_prob, k)

                f0, f1 = f_rot[0], f_rot[1]
                policy_vec = np.concatenate([b_rot.flatten(), [pass_prob]])

                augmented.append([[f0, f1], policy_vec, v])
                
                f_flip = np.flip(f_rot, axis=2)
                b_flip = np.fliplr(b_rot)

                f0, f1 = f_flip[0], f_flip[1]
                augmented.append(
                    ([f0, f1],
                    np.concatenate([b_flip.flatten(), [pass_prob]]),
                    v))

        return augmented
