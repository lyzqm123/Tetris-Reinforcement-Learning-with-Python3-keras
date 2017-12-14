import time
import numpy as np
from tetris import Env

EPISODE = 1000
GAME_VELOCTY = 0.05
ACTION_VELCOCITY = 0.01

if __name__ == "__main__":
    tetris = Env()

    start_time = time.time()
    action_time = time.time()
    epi = 0
    while epi < EPISODE:
        end_time = time.time()
        if end_time - action_time >= ACTION_VELCOCITY:
            tetris.step(np.random.randint(3))
            action_time = time.time()

        if end_time - start_time >= GAME_VELOCTY:
            if tetris.is_game_end():
                print('episode:{}, score:{}'.format(epi,tetris.score))
                tetris.reset()
                epi += 1

            else:
                tetris.step(0)
                start_time = time.time()