import tkinter as tk
import time
import numpy as np
import random
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
    def __init__(self, agent, environment):
        super(GridWorld, self).__init__()
        
        self.agent = agent
        self.environment = environment
        
        self.title('가치 이터레이션')
        
        self.geometry('{0}x{1}'.format(COLUMN*UNIT, ROW*UNIT+50))
        
        self.values = []
        self.arrows = []
        
        self.iteration_count = 0
        self.improvement_count = 0
        self.is_moving = 0
        
        (self.up, self.down, self.left,self.right), self.shapes = self.load_images()
        
        self.canvas = self.build_canvas()
        
        # 성공과 실패 칸에 보상 텍스트 쓰기
        self.write_text_reward(TARGET[0], TARGET[1], "R : 1.0")
        for b in BOMBS :
            self.write_text_reward(b[0], b[1], "R : -1.0")
            
            
            
    def build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                           height=ROW * UNIT,
                           width=COLUMN * UNIT)
        
        # 버튼 초기화
        def create_button( text, command, xr):
            button =tk.Button(self, text=text, command=command)
            button.configure(width=10, height=2, activebackground="#33B5E5")
            canvas.create_window(COLUMN*UNIT*xr, ROW*UNIT+25, window=button)
            return button
        
        create_button ("최적 가치 계산", self.calculate_value, 0.13)
        create_button ("정책 보기", self.print_optimal_policy, 0.37)
        create_button ("출발", self.move_by_policy, 0.62)
        create_button ("리셋", self.clear, 0.87)
        
        
        # 가로줄 그리기
        for col in range(0, ROW * UNIT, UNIT):
            x1, y1, x2, y2 = col, 0, col, COLUMN*UNIT
            canvas.create_line(x1, y1, x2, y2, fill='gray')
        
        # 세로줄 그리기   
        for row in range(0, COLUMN * UNIT, UNIT):
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


    def clear(self):

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
        elif action == 1 and location[0] < COLUMN - 1:  # 하
            base_action[1] += UNIT
        elif action == 2 and location[1] > 0:  # 좌
            base_action[0] -= UNIT
        elif action == 3 and location[1] < ROW - 1:  # 우
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

        self.is_moving = 1

        # 시작 상태로 에이전트를 이동한다.
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT/2-x, UNIT/2-y)
        
        x, y = self.find_rectangle()
        while len(self.agent.get_action([x,y])) != 0:
            action = random.sample(self.agent.get_action([x, y]), 1)[0]
            self.after(500, self.move_rectangle(action))
            x, y = self.find_rectangle()
        self.is_moving = 0


    # 최적 액션에 화살표 그려주기
    def draw_one_arrow(self, col, row, action):
        if col == 2 and row == 2:
            return
        if action == 0:  # 위
            x, y = 50+(UNIT*row), 10+(UNIT*col)
            self.arrows.append(self.canvas.create_image( x, y, image=self.up))
        
        elif action == 1:  # 아래
            x, y = 50+(UNIT*row), 90+(UNIT*col)
            self.arrows.append(self.canvas.create_image( x, y, image=self.down))
            
        elif action == 3:  # 오른쪽
            x, y= 90+(UNIT*row), 50+(UNIT*col)
            self.arrows.append(self.canvas.create_image( x, y, image=self.right))
            
        elif action == 2:  # 왼쪽
            x, y = 10+(UNIT*row), 50+(UNIT*col)
            self.arrows.append(self.canvas.create_image( x, y, image=self.left))


    def draw_from_values(self, state, action_list):
        i = state[0]
        j = state[1]
        for action in action_list:
            self.draw_one_arrow(i, j, action)


    def print_value_table(self, value_table):
        for i in range(ROW):
            for j in range(COLUMN):
                self.write_text_value(i, j, value_table[i][j])


    def render(self):
        time.sleep(0.1)
        self.canvas.tag_raise(self.rectangle)
        self.update()


    def calculate_value(self):
        self.iteration_count += 1
        for i in self.values:
            self.canvas.delete(i)
            
        self.agent.value_iteration()
        self.print_value_table(self.agent.value_table)

    
    def print_optimal_policy(self):
        
        self.improvement_count += 1
        
        for i in self.arrows:
            self.canvas.delete(i)
        
        for state in self.environment.get_all_states():
            action = self.agent.get_action(state)
            self.draw_from_values(state, action)





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
        state[0] = (0 if state[0] < 0 else ROW-1 if state[0] > ROW-1 else state[0])
        
        # 세로 좌표가 0보다 적거나 COLUMN을 넘었을 경우 처리 
        state[1] = (0 if state[1] < 0 else COLUMN-1 if state[1] > COLUMN-1 else state[1])
        
        return state


    def get_all_states(self):
        return self.all_state