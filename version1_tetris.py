import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image
from time import sleep

np.random.seed(1)
UNIT = 30  # 픽셀 수
HEIGHT = 20  # 그리드 세로
WIDTH = 8   # 그리드 가로
MID = (WIDTH / 2 - 1) * UNIT # 블록 시작 점
dy = [UNIT, 0, 0]       # 아래, 왼쪽, 오른쪽
dx = [0, UNIT, -UNIT]   
PLUS_SCORE = 10.0
basic_counter_str = 'Test : '
basic_score_str = 'Score : '


#   [x, y]

         # 네모 테트로미노
pos = [[[[MID, 0], [MID + UNIT, 0], [MID, 0 + UNIT], [MID + UNIT, 0 + UNIT]]],                      
         # 일직선 테트로미노
      [[[MID - UNIT * 2, 0], [MID - UNIT, 0], [MID, 0], [MID + UNIT, 0]],
       [[MID, UNIT * 2], [MID, UNIT], [MID, 0], [MID, -UNIT]]],                      
         # 'ㄴ' 모양 테트로미노
      [[[MID, -UNIT], [MID, 0], [MID, UNIT], [MID + UNIT, UNIT]],
       [[MID + UNIT, 0], [MID, 0], [MID - UNIT, 0], [MID - UNIT, UNIT]],
       [[MID, UNIT], [MID, 0], [MID, -UNIT], [MID - UNIT, -UNIT]],
       [[MID - UNIT, 0], [MID, 0], [MID + UNIT, 0], [MID + UNIT, -UNIT]]],                
         # 지그재그 모양 테트로미노
      [[[MID, -UNIT], [MID, 0], [MID + UNIT, 0], [MID + UNIT, UNIT]],
       [[MID + UNIT, 0], [MID, 0], [MID, UNIT], [MID - UNIT, UNIT]]],           
         # 'ㅜ' 모양 테트로미노
      [[[MID - UNIT, 0], [MID, 0], [MID + UNIT, 0], [MID, UNIT]],
       [[MID, -UNIT], [MID, 0], [MID, UNIT], [MID - UNIT, 0]],
       [[MID + UNIT, 0], [MID, 0], [MID - UNIT, 0], [MID, -UNIT]],
       [[MID, UNIT], [MID, 0], [MID, -UNIT], [MID + UNIT, 0]]],]    
         # 네모 테트로미노

#pos = [
#         # 'ㅜ' 모양 테트로미노
#      [[[MID - UNIT, 0], [MID, 0], [MID + UNIT, 0], [MID, UNIT]],
#       [[MID, -UNIT], [MID, 0], [MID, UNIT], [MID - UNIT, 0]],
#       [[MID + UNIT, 0], [MID, 0], [MID - UNIT, 0], [MID, -UNIT]],
#       [[MID, UNIT], [MID, 0], [MID, -UNIT], [MID + UNIT, 0]]]]    

class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.score = 0.0
        self.score_weight = []
        self.counter = 0
        for i in range(HEIGHT):
            if i <= 2:
                self.score_weight.append(0.0)
            else:
                self.score_weight.append((i-2)*(i-2)*0.00082976)

        self.color = ["red","blue","green","yellow","purple"]
        self.block_kind = len(pos)
        self.block = list()

        self.curr_block = np.random.randint(self.block_kind)
        self.curr_block_type = np.random.randint(len(pos[self.curr_block]))
        self.canvas, self.counter_board, self.score_board = self._build_canvas()
        self.map = [[0]*WIDTH for _ in range(HEIGHT)]


    #def _get_curr_block_pos(self):
    #    ret = []
    #    for n in range(4):
    #        s = (self.canvas.coords(self.block[n]))
    #        y = int(s[1] / UNIT)
    #        x = int(s[0] / UNIT)
    #        ret.append([y,x])
    #    return ret

    def _clear_map(self):
        for n in range(HEIGHT):
            for m in range(WIDTH):
                self.map[n][m] = 0

    def _erase_down_canvas(self, iy):
        for crect in self.canvas.find_withtag("rect"):
            if int(self.canvas.coords(crect)[1]) == iy*UNIT:
                self.canvas.delete(crect)

    def _move_all_canvas_down(self, iy):
        for crect in self.canvas.find_withtag("rect"):
            if int(self.canvas.coords(crect)[1]) < iy*UNIT:
                self.canvas.move(crect, 0, UNIT)

    def _add_canvas(self):
        pos = self.make_block()
        rect1 = self.canvas.create_rectangle(pos[0][0], pos[0][1], pos[0][0] + UNIT,
                                             pos[0][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")
        rect2 = self.canvas.create_rectangle(pos[1][0], pos[1][1], pos[1][0] + UNIT,
                                             pos[1][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")
        rect3 = self.canvas.create_rectangle(pos[2][0], pos[2][1], pos[2][0] + UNIT,
                                             pos[2][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")
        rect4 = self.canvas.create_rectangle(pos[3][0], pos[3][1], pos[3][0] + UNIT,
                                             pos[3][1] + UNIT, fill=self.color[self.curr_block],
                                             tag="rect")

        self.block = [rect1, rect2, rect3, rect4]
        self.canvas.pack()

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='black',
                           height=HEIGHT * UNIT,
                           width=(WIDTH + 7)* UNIT)
        
        # 점수 배경 판
        canvas.create_rectangle(275, 150, 425,250, fill='dim gray')    
        canvas.create_text(345, 590,
                           font = "Times 11 bold",
                           fill = 'white',
                           text = "2018.   Developed by 'lyzqm'")

        counter_board = canvas.create_text(350, 175,
                           fill = "gray22",
                           font = "Times 13 bold",
                           text = basic_counter_str + str(int(self.counter)))

        score_board = canvas.create_text(350, 205,
                           fill = "gray22",
                           font = "Times 13 bold",
                           text = basic_score_str + str(int(self.score)))


        # 그리드 생성
        for c in range(0, (WIDTH + 1)* UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1, fill = 'white')
        
        for r in range(0, HEIGHT * UNIT, UNIT):      # 0~400 by 80
            x0, y0, x1, y1 = 0, r, WIDTH * UNIT, r
            canvas.create_line(x0, y0, x1, y1, fill = 'white')
        

        # 캔버스에 이미지 추가
        pos = self.make_block()
        rect1 = canvas.create_rectangle(pos[0][0], pos[0][1], pos[0][0] + UNIT,
                                        pos[0][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        rect2 = canvas.create_rectangle(pos[1][0], pos[1][1], pos[1][0] + UNIT,
                                        pos[1][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        rect3 = canvas.create_rectangle(pos[2][0], pos[2][1], pos[2][0] + UNIT,
                                        pos[2][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        rect4 = canvas.create_rectangle(pos[3][0], pos[3][1], pos[3][0] + UNIT,
                                        pos[3][1] + UNIT, fill=self.color[self.curr_block],
                                        tag="rect")
        self.block = [rect1,rect2,rect3,rect4]
        canvas.pack()
        return canvas,counter_board,score_board

    def make_block(self):
        return pos[self.curr_block][self.curr_block_type]

    def reset(self):
        self.score = 0.0
        self.counter += 1
        self.canvas.itemconfigure(self.counter_board,
                                  text=basic_counter_str + str(int(self.counter)))
        self.canvas.itemconfigure(self.score_board,
                                  text=basic_score_str + str(int(self.score)))
        self.update()
        self.canvas.delete("rect")
        self._clear_map()
        self._add_canvas()

    def step(self, action):
        self.render()
       
        reward = float
        if action < 3:
            reward = self.move(action)
        else:
            reward = self.rotate(action)
        if reward >= 0.00000:
            # make new block!
            self.curr_block = np.random.randint(self.block_kind)
            self.curr_block_type = np.random.randint(len(pos[self.curr_block]))
            self._add_canvas()
        self.canvas.tag_raise(self.block)
        if reward < 0.00000:
            return 0.0
        else:
            return reward



    def possible_to_move(self, action):
        for n in range(len(self.block)):
            s = self.canvas.coords(self.block[n])
            y = s[1] + dy[action]
            x = s[0] + dx[action]

            # 범위밖 - stay
            if x >= WIDTH * UNIT or x < 0:
                return 1
            ny = int(y/UNIT)
            nx = int(x/UNIT)

            # 마지막줄 - add canvas
            if y >= HEIGHT * UNIT:
                return 2
            if self.map[ny][nx] == 1:
                if action == 0:
                    return 2
                else:
                    return 1

        # 이동가능함 - move
        return 3

    def is_map_horizon(self):
        for n in range(HEIGHT - 1, 0, -1):
            cnt = 0
            for m in range(WIDTH):
                if self.map[n][m] != 1:
                    break
                cnt += 1
            if cnt == WIDTH:
                return n
        return -1

    def move(self, action):
        ret = 0.0
        base_action = np.array([0, 0])
        flag = self.possible_to_move(action)

        # 해당 자리에 고정시켜줌
        if flag == 2:
            for n in range(4):
                s = (self.canvas.coords(self.block[n]))
                y = int(s[1] / UNIT)
                x = int(s[0] / UNIT)
                self.score += self.score_weight[y]
                ret += self.score_weight[y]
                self.map[y][x] = 1

            # 한줄이 꽉차있으면 비워주고 점수를 더해줌
            break_cnt = 0
            while True:
                y = self.is_map_horizon()
                if y == -1:
                    break
                self._erase_down_canvas(y)
                self._move_all_canvas_down(y)
                break_cnt += 1
                for m in range(WIDTH):
                    for n in range(y , 2, -1):
                        self.map[n][m] = self.map[n-1][m]
            self.score += PLUS_SCORE * break_cnt
            ret += PLUS_SCORE * break_cnt
            self.canvas.itemconfigure(self.score_board,
                                      text = basic_score_str + str(int(self.score)))
            return ret

        # move
        elif flag == 3:
            base_action[1] += dy[action]
            base_action[0] += dx[action]
            for n in range(4):
                self.canvas.move(self.block[n], base_action[0], base_action[1])

        #self.canvas.coords(self.block)
        return -1.0

    def possible_to_rotate(self, next_block):
        for i in range(len(self.block)):
            y = int(next_block[i][1] / UNIT)
            x = int(next_block[i][0] / UNIT)
            if y < 0 or y >= HEIGHT or x < 0 or x >= WIDTH or self.map[y][x] == 1:
                return False
        return True

    def rotate(self, dir):
        ret = 0.0

        dir = (1 if dir == 3 else 3)
        next_block = [[0]*2 for _ in range(len(self.block))]
        curr_size = len(pos[self.curr_block])
        for i in range(len(self.block)):
            s = self.canvas.coords(self.block[i])
            # y
            next_block[i][1] = s[1] + pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][1] - pos[self.curr_block][self.curr_block_type][i][1]
            # x
            next_block[i][0] = s[0] + pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][0] - pos[self.curr_block][self.curr_block_type][i][0]
            
        if self.possible_to_rotate(next_block) == False:
            for i in range(len(self.block)):
                s = self.canvas.coords(self.block[i])
            return -1.

        for i in range(len(self.block)):
            self.canvas.move(self.block[i], 
                             pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][0] - pos[self.curr_block][self.curr_block_type][i][0], 
                             pos[self.curr_block][(self.curr_block_type + dir) % curr_size][i][1] - pos[self.curr_block][self.curr_block_type][i][1])
        self.curr_block_type = (self.curr_block_type + dir) % curr_size
        return -1.
        

    def is_game_end(self):
        for n in range(3):
            for m in range(WIDTH):
                if self.map[n][m] == 1:
                    return True
        return False

    def render(self):
        # 게임 속도 조정
        self.update()