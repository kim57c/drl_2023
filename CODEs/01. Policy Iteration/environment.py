import tkinter as tk # tkinter 설치 필요
from tkinter import Button
import time
import numpy as np
from PIL import ImageTk, Image

PhotoImage = ImageTk.PhotoImage


UNIT = 100  # 각 상태별 픽셀 수

ROW = 5  # 그리드월드 열의 수  
COLUMN = 5  # 그리드월드 행의 수 

ACTIONS = [0, 1, 2, 3]  # 좌, 우, 상, 하
ACTIONS_COORDS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 액션에 대한 이동 좌표

TARGET = [2,2]
BOMBS =[[1,2], [2,1]] # , [3,1] 


class GridWorld(tk.Tk):
    
    def __init__(self, agent):
        
        super(GridWorld, self).__init__()
        self.agent = agent
        
        self.title('정책 이터레이터')
        
        # 화면의 크기 설정
        self.geometry('{0}x{1}'.format(COLUMN*UNIT, ROW*UNIT+50))
        
        self.values = []
        self.arrows = []
        
        self.evaluation_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        
        (self.up, self.down, self.left, self.right), self.shapes = self.load_images()
        
        self.canvas = self.build_canvas()
        
        # 성공과 실패 칸에 보상 텍스트 쓰기
        self.write_text_reward(TARGET[0], TARGET[1], "R : 1.0")
        for b in BOMBS :
            self.write_text_reward(b[0], b[1], "R : -1.0")
            


    def build_canvas(self):
        
        canvas = tk.Canvas(self, bg='white', height=ROW*UNIT, width=COLUMN*UNIT)
        
        # 버튼 초기화
        def create_button( text, command, xr):
            button = Button(self, text=text, command=command)
            button.configure(width=10, height=2, activebackground="#33B5E5")
            canvas.create_window(COLUMN*UNIT*xr, ROW*UNIT+25, window=button)
            return button
        
        create_button ("정책 평가", self.evaluate_policy, 0.13)
        create_button ("정책 개선", self.improve_policy, 0.37)
        create_button ("출발", self.move_by_policy, 0.62)
        create_button ("리셋", self.reset, 0.87)
        
        # 가로줄 그리기
        for col in range(0, ROW * UNIT, UNIT):  # 0~400 by 80
            x1, y1, x2, y2 = col, 0, col, COLUMN*UNIT
            canvas.create_line(x1, y1, x2, y2, fill='gray')
        
        # 세로줄 그리기   
        for row in range(0, COLUMN * UNIT, UNIT):  # 0~400 by 80
            x1, y1, x2, y2 = 0, row, ROW*UNIT, row
            canvas.create_line(x1, y1, x2, y2, fill='gray')

        # 에이전트 이미지
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        
        # 성공 실패 이미지
        canvas.create_image( TARGET[0]*UNIT+UNIT/2, TARGET[1]*UNIT+UNIT/2, image=self.shapes[2])
        for b in BOMBS :
            canvas.create_image( b[1]*UNIT+UNIT/2, b[0]*UNIT+UNIT/2, image=self.shapes[1])
        
        canvas.pack()

        return canvas


    def load_images(self):
        up = PhotoImage(Image.open("./img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("./img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("./img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("./img/down.png").resize((13, 13)))
        rectangle = PhotoImage(Image.open("./img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("./img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("./img/circle.png").resize((65, 65)))
        return (up, down, left, right), (rectangle, triangle, circle)


    def reset(self):
        
        if self.is_moving != 0:
            return 
        
        self.evaluation_count = 0
        self.improvement_count = 0
        
        for i in self.values:
            self.canvas.delete(i)

        for i in self.arrows:
            self.canvas.delete(i)
        
        self.agent.reset ()
        
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT/2-x, UNIT/2-y)


    def write_text_value(self, row, col, contents, font='Avenir', size=11,
                   style='normal', anchor="nw"):
        
        origin_x, origin_y = 85, 70
        x, y = origin_y + (UNIT * col), origin_x + (UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        return self.values.append(text)


    def write_text_reward(self, row, col, contents, font='Avenir', size=11,
                    style='normal', anchor="nw"):
        
        origin_x, origin_y = 5, 5
        x, y = origin_y + (UNIT*col), origin_x+(UNIT * row)
        font = (font, str(size), style)
        text = self.canvas.create_text(x, y, fill="black", text=contents,
                                       font=font, anchor=anchor)
        
        # return self.values.append(text)


    def move_rectangle(self, action):
        
        base_action = np.array([0, 0])
        location = self.find_rectangle()
        
        self.render()
        
        if action == 0 and location[0] > 0:  # 상
            base_action[1] -= UNIT
        elif action == 1 and location[0] < ROW-1:  # 하
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0:  # 좌
            base_action[0] -= UNIT
        elif action == 3 and location[1] < COLUMN-1:  # 우
            base_action[0] += UNIT
        
        # 에이전트 이동 
        self.canvas.move(self.rectangle, base_action[0], base_action[1])



    # 현재 에이전트 사각형의 상태 좌표룰 찾아준다.
    def find_rectangle(self):
        # 에이전트 사각형 이지미의 가운데 좌표를 반환
        coords = self.canvas.coords(self.rectangle)
        x = (coords[0]/100)-0.5
        y = (coords[1]/100)-0.5
        return int(y), int(x)

    
    def move_by_policy(self):
        
        if self.improvement_count == 0 or self.is_moving == 1:
            return
        
        self.is_moving = 1

        # 시작 상태로 에이전트를 이동한다.
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT/2-x, UNIT/2-y)
        
        
        x, y = self.find_rectangle()
        while len(self.agent.policy_table[x][y]) != 0:
            self.after(500, self.move_rectangle(self.agent.get_action([x, y])))
            x, y = self.find_rectangle()
        self.is_moving = 0


    def draw_one_arrow(self, row, col, policy):
        if col == 2 and row == 2:
            return

        if policy[0] > 0:  # 위
            origin_x, origin_y = 50+(UNIT*col), 10+(UNIT*row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.up))
            
        if policy[1] > 0:  # 아래
            origin_x, origin_y = 50+(UNIT*col), 90+(UNIT*row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.down))
            
        if policy[2] > 0:  # 왼쪽
            origin_x, origin_y = 10+(UNIT*col), 50+(UNIT*row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.left))
            
        if policy[3] > 0:  # 오른쪽
            origin_x, origin_y = 90+(UNIT*col), 50+(UNIT*row)
            self.arrows.append(self.canvas.create_image(origin_x, origin_y, image=self.right))


    def draw_from_policy(self, policy_table):
        for i in range(ROW):
            for j in range(COLUMN):
                self.draw_one_arrow(i, j, policy_table[i][j])

    def print_value_table(self, value_table):
        for i in range(ROW):
            for j in range(COLUMN):
                self.write_text_value(i, j, value_table[i][j])

    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.rectangle)
        self.update()


    def evaluate_policy(self):
        self.evaluation_count += 1
        for i in self.values:
            self.canvas.delete(i)
        self.agent.policy_evaluation()
        self.print_value_table(self.agent.value_table)


    def improve_policy(self):
        self.improvement_count += 1
        for i in self.arrows:
            self.canvas.delete(i)
        self.agent.policy_improvement()
        self.draw_from_policy(self.agent.policy_table)



class Enveronment:
    
    def __init__(self):
        
        self.reward = [[0] * ROW for _ in range(COLUMN)]
        
        
        self.reward[TARGET[0]][TARGET[1]] = 1 # 성공적 도착 보상 1 
        for b in BOMBS :
            self.reward[b[0]][b[1]] = -1  # 지뢰 보상 -1
         
        self.all_state = []

        for x in range(ROW):
            for y in range(COLUMN):
                state = [x, y]
                self.all_state.append(state)


    # 어떤 상태에서 특정 액션시의 보상을 반환
    def get_reward(self, state, action):
        next_state = self.state_after_action(state, action)
        return self.reward[next_state[0]][next_state[1]]


    # 액션후의 상태를 반환
    def state_after_action(self, state, action_index):
        action = ACTIONS_COORDS[action_index]
        return self.check_boundary([state[0] + action[0], state[1] + action[1]])


    @staticmethod
    def check_boundary(state):
        
        # 가로 좌표가 0보다 적거나 ROW를 넘었을 경우 처리 
        state[0] = (0 if state[0] < 0 else ROW - 1 if state[0] > ROW - 1 else state[0])
        
        # 세로 좌표가 0보다 적거나 COLUMN을 넘었을 경우 처리 
        state[1] = (0 if state[1] < 0 else COLUMN - 1 if state[1] > COLUMN - 1 else state[1])
        
        return state


    def get_all_states(self):
        return self.all_state