import time
import numpy as np
import tkinter as tk # tkinter 설치 필요
from PIL import ImageTk, Image
import time

# np.random.seed(1)
PhotoImage = ImageTk.PhotoImage

UNIT = 100 # 각 상태별 픽셀 수
HEIGHT = 5 # 그리드월드 높이
WIDTH = 5  # 그리드월드 폭

TARGET = [2,2]
BOMBS =[[1,2], [2,1]]
# BOMBS =[[1,2], [2,1], [3,1]]



class Environment(tk.Tk):
    
    def __init__(self):
    
        super(Environment, self).__init__()
        
        self.title('Q-러닝')
        self.geometry('{0}x{1}'.format(WIDTH*UNIT, HEIGHT*UNIT+50))
        
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        
        self.is_moving = 0
        self.values = []

        # 성공 칸에 보상 텍스트 쓰기
        self.write_text (TARGET[0], TARGET[1], "R : 100")
        
        # 실패 칸에 보상 텍스트 쓰기
        [self.write_text(b[0], b[1], "R : -100") for b in BOMBS]
        
        
        
    def load_images(self):
        rectangle = PhotoImage(Image.open("./img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(Image.open("./img/triangle.png").resize((65, 65)))
        circle = PhotoImage(Image.open("./img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle



    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white', height=HEIGHT*UNIT, width=WIDTH*UNIT)
        
        # 버튼 초기화
        def create_button( text, command, xr):
            button =tk.Button(self, text=text, command=command)
            button.configure(width=10, height=2, activebackground="#33B5E5")
            canvas.create_window(WIDTH*UNIT*xr, HEIGHT*UNIT+25, window=button)
            return button
        
        create_button ("에피소드 1번", self.episode_one, 0.13)
        create_button ("에피소드 100번", self.episode_many, 0.37)
        create_button ("출발", self.start_simul, 0.62)
        create_button ("리셋", self.clear, 0.87)
        
        def draw_line(x0, y0, x1, y1):
            canvas.create_line(x0, y0, x1, y1, fill='gray')
        
        # 가로줄 그리기
        [draw_line(c, 0, c, HEIGHT*UNIT) for c in range(0, WIDTH*UNIT, UNIT)]
        # 세로줄 그리기
        [draw_line(0, r, WIDTH*UNIT, r) for r in range(0, HEIGHT*UNIT, UNIT)]

        # 이미지 생성 
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])
        
        self.triangles = []
        def create_triangle(b):
            triangle = canvas.create_image(b[0]*UNIT+50, b[1]*UNIT+50, image=self.shapes[1])
            self.triangles.append(triangle )
        [create_triangle(b) for b in BOMBS]
        
        self.circle = canvas.create_image(250, 250, image=self.shapes[2])

        # 캔버스 팩
        canvas.pack()

        return canvas

    
    
    def episode_one(self):
        self.agent.episode()

            
    def episode_many(self):
        [self.agent.episode(False) for _ in range(100)]
        

    def clear(self):

        if self.is_moving != 0:
            return 
        
        [self.canvas.delete(i) for i in self.values]
        
        self.agent.reset ()
        
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT/2-x, UNIT/2-y)


    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0]-50)/100)
        y = int((coords[1]-50)/100)
        return [x, y]


    def start_simul(self):

        self.is_moving = 1

        # 시작 상태로 에이전트를 이동한다.
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT/2-x, UNIT/2-y)
        self.redraw()
        x, y = self.find_rectangle()
        
        def is_close(state):
            return True if (state==TARGET or state in BOMBS) else False
        
        while not is_close([x, y]):
            time.sleep(0.5)
            action_values = self.agent.get_state_actions([x, y])
            action = self.agent.arg_max(action_values)
            self.move_rectangle(action)
            x, y = self.find_rectangle()
            
        self.is_moving = 0



    def move_rectangle(self, action):
        
        base_action = np.array([0, 0])
        location = self.find_rectangle()
        
        if action == 0 and location[1] > 0:  # 상
            base_action[1] -= UNIT
        elif action == 1 and location[1] < HEIGHT - 1:  # 하
            base_action[1] += UNIT
        elif action == 2 and location[0] > 0:  # 좌
            base_action[0] -= UNIT
        elif action == 3 and location[0] < WIDTH - 1:  # 우
            base_action[0] += UNIT
            
        # 에이전트 이동 
        self.canvas.move(self.rectangle, base_action[0], base_action[1])

        self.redraw()
    
    
    def write_q_table(self, q_table):
        
        [self.canvas.delete(i) for i in self.values]
        
        def write_value (state, actions ):
            
            state = state[1:len(state)-1]
            idx = state.index(',')
            x, y = int(state[0:idx]), int(state[idx+1:len(state)])
            if [x,y] in BOMBS or [x,y] == TARGET :
                return 
            
            def write_fn(value, gx, gy ):
                self.write_text(x, y, value, gx=gx, gy=gy, 
                                color='blue', list=self.values )
            
            coords = [[40,12],[50,92],[17,45],[80,55]]
            for key, value in actions.items():
                coord =coords[int(key)]
                write_fn (round(value,2), gx=coord[0], gy=coord[1] )
            
        [write_value(state, value) for state, value in q_table.items()]
        
                
    
    def write_text(self, x, y, contents, gx=23, gy=9, color='black', list=None ):
        
        x, y = gx+(UNIT*x), gy+(UNIT*y)
        font = ('Avenir', str(12), 'normal')
        text = self.canvas.create_text(x, y, fill=color, text=contents, font=font ) 
        
        if list != None :
            list.append(text)
        
        
    def reset(self):
        self.update()
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT/2-x, UNIT/2-y)

        return self.coords_to_state(self.canvas.coords(self.rectangle))


    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])

        if action == 0 and state[1]>UNIT :  # 위
            base_action[1] -= UNIT
        elif action == 1 and state[1]<(HEIGHT-1)*UNIT:  # 아래
            base_action[1] += UNIT
        elif action == 2 and state[0]>UNIT:  # 왼쪽
            base_action[0] -= UNIT
        elif action == 3 and state[0]<(WIDTH-1)*UNIT :  # 오른쪽
            base_action[0] += UNIT
        
        # 에이전트 이동 
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        
        # 캔버스의 맨 앞으로 사각형 이동(보이기 위해)
        self.canvas.tag_raise(self.rectangle)

        next_coord = self.canvas.coords(self.rectangle)
        next_state = self.coords_to_state(next_coord)
        
        # 보상 함수
        if next_state == TARGET:
            reward = 100
            done = True
            
        elif next_state in BOMBS:
            reward = -100
            done = True
            
        else:
            reward = 0
            done = False
            
        return next_state, reward, done


    # 현재 에이전트 사각형의 상태 좌표룰 찾아준다. 사각형 이지미의 가운데 좌표를 반환
    def find_rectangle(self):
        coords = self.canvas.coords(self.rectangle)
        x = (coords[0]/100)-0.5
        y = (coords[1]/100)-0.5
        return int(x), int(y)


    def redraw(self):
        self.update()
