import time
import numpy as np
from version1_tetris import Env

EPISODE = 50000000
GAME_VELOCTY = 0.0000001               # 해당 시간마다 테트로미노 하강
ACTION_VELCOCITY = 0.0000001            # 해당 시간마다 테트로미노 action 취할 수 있음
ACTION_SIZE = 5         # 아래, 왼쪽, 오른쪽, 시계방향, 반시계방향

if __name__ == "__main__":
    tetris = Env()

    start_time = time.time()
    action_time = time.time()
    epi = 0
    count = 0
    while epi < EPISODE:
        end_time = time.time()

        if end_time - action_time >= ACTION_VELCOCITY:
            tetris.step(np.random.randint(5))
            action_time = time.time()

        if end_time - start_time >= GAME_VELOCTY:
            if tetris.is_game_end():
                print('episode:{}, score:{}'.format(epi,tetris.score))
                if tetris.score >= 10:
                    count+=1
    #                print('epi : {}, probabaility :{}'.format(epi, count * 100. / (epi + 1)))
                tetris.reset()
                epi += 1

            else:
                tetris.step(0)
                start_time = time.time()