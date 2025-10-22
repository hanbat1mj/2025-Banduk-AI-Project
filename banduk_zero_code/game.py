
import random
import math
import more_itertools as mit

class State:
    
    def __init__(self, pieces=None, enemy_pieces=None, r_pieces=None, d_pieces=None,
                id_pieces=None, stone_id=100000, depth=0):
        
        # 연속 패스에 따른 종료
        self.pass_end = False

        # 돌의 배치
        self.pieces = pieces
        self.enemy_pieces = enemy_pieces
        self.r_pieces = r_pieces
        self.d_pieces = d_pieces
        self.id_pieces = id_pieces
        self.stone_id = stone_id
        self.depth = depth

        # 돌의 초기 배치
        if pieces == None or enemy_pieces == None:
            self.pieces = [0] * 36
            self.enemy_pieces = [0] * 36
            self.r_pieces = [0] * 36
            self.d_pieces = [0] * 36
            self.id_pieces = [0] * 36
    
        # 돌의 수 얻기
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count
    
    # 패배 여부 판정
    def is_lose(self):
        result_element = None
        if self.is_first_player():
            result_element = self.piece_count(self.pieces) < self.piece_count(self.enemy_pieces) + 0.5
        else:
            result_element = self.piece_count(self.pieces) + 0.5 < self.piece_count(self.enemy_pieces)
        return self.is_done() and result_element

    # 무승부 여부 판정
    def is_draw(self):
        return self.is_done() and self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)

    # 게임 종료 여부 판정
    def is_done(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 36 or self.pass_end
    
        # 다음 상태 얻기
    def next(self, action):
        state = State(self.pieces.copy(), self.enemy_pieces.copy(), self.r_pieces.copy(), self.d_pieces.copy(),
                      self.id_pieces.copy(), self.stone_id, self.depth)
        if action != 36:
            state.is_legal_action_xy(action % 6, int(action / 6), True)
        w = state.pieces
        state.pieces = state.enemy_pieces
        state.enemy_pieces = w
        state.depth += 1
        # 2회 연속 패스 판정
        if action == 36 and state.legal_actions() == [36]:
            state.pass_end = True
        return state
    
        # 합법적인 수 리스트 얻기
    def legal_actions(self):
        actions = []
        for j in range(0, 6):
            for i in range(0, 6):
                if self.is_legal_action_xy(i, j):
                    actions.append(i + j * 6)
        if len(actions) == 0:
            actions.append(36)  # 패스
        return actions

    # 임의의 매스가 합법적인 수인지 판정
    def is_legal_action_xy(self, x, y, flip=False):
        
        def calculate_flip(x, y, flip):
            def is_neg(num):
                return abs(num) != num
            
            if self.is_first_player():
                stone_score = 0
                stone_id = self.stone_id
                self.stone_id += 1
                p1 = y - 1
                p2 = y + 1
                p3 = x + 1
                p4 = x - 1
                b1 = False if (p1 == -1) else True
                b2 = False if (p2 == 6) else True
                b3 = False if (p3 == 6) else True
                b4 = False if (p4 == -1) else True
                around_stone_id_map = {}
                _id = None
                _place = None
                _id_opponent = None
                opponent_number = 0
                wall_number = 0
                stone_number = 0
                if (b1):
                    _id = self.id_pieces[x + p1 * 6]
                    _place = self.r_pieces[x + p1 * 6]
                    if (_id != 0):
                        if not is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                if (b2):
                    _id = self.id_pieces[x + p2 * 6]
                    _place = self.r_pieces[x + p2 * 6]
                    if (_id != 0):
                        if not is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                if (b3):
                    _id = self.id_pieces[p3 + y * 6]
                    _place = self.r_pieces[p3 + y * 6]
                    if (_id != 0):
                        if not is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                if (b4):
                    _id = self.id_pieces[p4 + y * 6]
                    _place = self.r_pieces[p4 + y * 6]
                    if (_id != 0):
                        if not is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                
                if (stone_number == 0):
                    if (wall_number + opponent_number == 4):
                        return False
                    if (flip):
                        self.pieces[x + y * 6] = 1
                        self.r_pieces[x + y * 6] = 1
                        self.id_pieces[x + y * 6] = stone_id
                        self.d_pieces = [0] * 36
                else:
                    if (flip):
                        stone_score = 0
                        for k in around_stone_id_map.keys():
                            stone_score += around_stone_id_map[k]
                            self.id_pieces = [stone_id if i==k else i for i in self.id_pieces]
                        stone_score += 1
                        
                        self.pieces[x + y * 6] = 1
                        self.r_pieces[x + y * 6] = stone_score
                        self.id_pieces[x + y * 6] = stone_id
                        self.d_pieces = [0] * 36

                        _list = list(mit.locate(self.id_pieces, lambda x: x == stone_id))

                        for i in _list:
                            self.r_pieces[i] = stone_score
                            self.id_pieces[i] += 200
                if (opponent_number == 1):
                    if (flip):
                        _list = list(mit.locate(self.id_pieces, lambda x: x == _id_opponent))

                        if (self.r_pieces[x + y * 6] > abs(self.r_pieces[_list[0]]) + 1):
                            for i in _list:
                                self.enemy_pieces[i] = 0
                                self.r_pieces[i] = 0
                                self.d_pieces[i] = 1
                                self.id_pieces[i] = 0
            else:
                stone_score = 0
                stone_id = -self.stone_id
                self.stone_id += 1
                p1 = y - 1
                p2 = y + 1
                p3 = x + 1
                p4 = x - 1
                b1 = False if (p1 == -1) else True
                b2 = False if (p2 == 6) else True
                b3 = False if (p3 == 6) else True
                b4 = False if (p4 == -1) else True
                around_stone_id_map = {}
                _id = None
                _place = None
                _id_opponent = None
                opponent_number = 0
                wall_number = 0
                stone_number = 0
                if (b1):
                    _id = self.id_pieces[x + p1 * 6]
                    _place = self.r_pieces[x + p1 * 6]
                    if (_id != 0):
                        if is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                if (b2):
                    _id = self.id_pieces[x + p2 * 6]
                    _place = self.r_pieces[x + p2 * 6]
                    if (_id != 0):
                        if is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                if (b3):
                    _id = self.id_pieces[p3 + y * 6]
                    _place = self.r_pieces[p3 + y * 6]
                    if (_id != 0):
                        if is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                if (b4):
                    _id = self.id_pieces[p4 + y * 6]
                    _place = self.r_pieces[p4 + y * 6]
                    if (_id != 0):
                        if is_neg(_place):
                            around_stone_id_map[_id] = _place
                            stone_number += 1
                        else:
                            _id_opponent = _id
                            opponent_number += 1
                else:
                    wall_number += 1
                
                if (stone_number == 0):
                    if (wall_number + opponent_number == 4):
                        return False
                    if (flip):
                        self.pieces[x + y * 6] = 1
                        self.r_pieces[x + y * 6] = -1
                        self.id_pieces[x + y * 6] = stone_id
                        self.d_pieces = [0] * 36
                else:
                    if (flip):
                        stone_score = 0
                        for k in around_stone_id_map.keys():
                            stone_score += around_stone_id_map[k]
                            self.id_pieces = [stone_id if i==k else i for i in self.id_pieces]
                        stone_score -= 1
                        
                        self.pieces[x + y * 6] = 1
                        self.r_pieces[x + y * 6] = stone_score
                        self.id_pieces[x + y * 6] = stone_id
                        self.d_pieces = [0] * 36

                        _list = list(mit.locate(self.id_pieces, lambda x: x == stone_id))

                        for i in _list:
                            self.r_pieces[i] = stone_score
                            self.id_pieces[i] -= 200
                if (opponent_number == 1):
                    if (flip):
                        _list = list(mit.locate(self.id_pieces, lambda x: x == _id_opponent))

                        if (abs(self.r_pieces[x + y * 6]) > self.r_pieces[_list[0]]):
                            for i in _list:
                                self.enemy_pieces[i] = 0
                                self.r_pieces[i] = 0
                                self.d_pieces[i] = 1
                                self.id_pieces[i] = 0
            return True

        # 빈칸 없음
        if self.enemy_pieces[x + y * 6] == 1 or self.pieces[x + y * 6] == 1:
            return False

        if self.d_pieces[x + y * 6] == 1:
            return False

        # 돌을 놓음
        return calculate_flip(x, y, flip)

    # 선 수 여부 확인
    def is_first_player(self):
        return self.depth % 2 == 0

    # 문자열 표시
    def __str__(self):
        ox = ('o', 'x') if self.is_first_player() else ('x', 'o')
        str = ''
        for i in range(36):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            if i % 6 == 5:
                str += '\n'
        return str
    
    def feature(self):
        return [self.pieces, self.enemy_pieces]


# 랜덤으로 행동 선택
def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0, len(legal_actions) - 1)]


# 동작 확인
if __name__ == '__main__':
    # 상태 생성
    state = State()

    # 게임 종료 시까지 반복
    while True:
        # 게임 종료 시
        if state.is_done():
            break
        
        # 다음 상태 얻기
        state = state.next(random_action(state))

        # 문자열 출력
        print(state)
        print()
