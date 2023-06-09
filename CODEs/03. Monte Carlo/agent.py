import numpy as np
import random
import time

from environment import Environment
from environment import HEIGHT, WIDTH
ACTIONS = [0,1,2,3]


class Agent:
    def __init__(self):
        
        self.learning_rate = 0.01 # 학습률
        self.discount_factor = 0.9 # 할인율 
        self.epsilon = 0.1 # 입실론

        self.reset()
        
        
    def reset (self):
        # 가치함수 테이블 초기화
        self.value_table = {}
        
    
    # 환경에서 종료를 줄 때까지 한번의 에피소드를 수행한다.    
    def episode (self, show=True):
        
        samples = []
        done = False
        
        # 환경을 초기화
        state = self.environment.reset()
        action = self.get_action(state)
        
        # 에피소드가 종료될 때까지 
        while not done:
            
            # 이동을 그리기 위해
            if show :
                self.environment.redraw()
                time.sleep(0.3)
                
            # 현재 상태에서 액션 구하기
            action = self.get_action(state)

            # 환경에 다음 스텝을 진행하고, 다음 상태, 보상, 종료여부를 반환 받는다.
            next_state, reward, done = self.environment.step(action)
            
            samples.append([next_state, reward, done])
            
            state = next_state 

        if show :
            self.environment.redraw()    
        
        # 샘플링된 에피소드로 MC 업데이트
        self.update(samples)

        self.environment.write_value_table ( self.value_table )

    
    
    # 각 에피소드에서 방문한 상태들의 가치함수를 업데이트 한다.
    def update(self, samples):
        G_t = 0
        visit_state = []
        
        # 종료에서부터 계산하기 위해 리버스(리턴 G_t의 계산을 수월하게 하기 위해)
        for (state,reward,done) in reversed(samples):
            
            state = str(state) # 리스트를 키로 하기 위해 문자열로 
            
            # 한번 방문한 상태는 업데이트 하지 않는다.
            if state not in visit_state:
                
                visit_state.append(state)
                
                # 다음리턴 = 할인률*(보상+현재리턴)
                G_t = self.discount_factor*(reward+G_t)
                
                # 현재상태가치함수
                value = self.value_table[state] if state in self.value_table else 0.0
                
                # 다음상태가치함수=현재상태가치함수+학습률*(다음리턴-현재상태가치함수)
                self.value_table[state] = value+self.learning_rate*(G_t-value)

        
        
    # 랜덤값이 입실론보다 작으면 랜덤으로하고, 그렇지 않으면 최대 가치함수로 선택(입실론 그리디 정책) 
    def get_action(self, state):
        
        if np.random.random() < (1 -  self.epsilon):
            action_values = self.possible_next_action_values(state)
            action = self.arg_max(action_values)
        else:
            action = np.random.choice(ACTIONS)

        return int(str(action))


    # 가능한 다음 상태 반환
    def possible_next_action_values(self, state):
        col, row = state
        next_action_value = [0.0] * 4

        if row == 0 : # 가장 윗쪽 열이면 다음 액션에 윗방향은 제외
            next_action_value[0] = -9999
        else :
            key = str([col,row-1])
            next_action_value[0] = self.value_table[key] if key in self.value_table else 0.0
        
        if row == HEIGHT-1 : # 가장 아랫쪽 열이면 다음 액션에 아랫방향은 제외
            next_action_value[1] = -9999
        else :
            key = str([col,row+1]) 
            next_action_value[1] = self.value_table[key] if key in self.value_table else 0.0
            
        if col == 0 : # 가장 왼쪽 행이면 다음 액션에 왼쪽방향은 제외
            next_action_value[2] = -9999
        else :
            key = str([col-1,row])
            next_action_value[2] = self.value_table[key] if key in self.value_table else 0.0
            
        if col == WIDTH-1 : # 가장 오른쪽 행이면 다음 액션에 오른쪽방향은 제외
            next_action_value[3] = -9999
        else :
            key = str([col+1,row])
            next_action_value[3] = self.value_table[key] if key in self.value_table else 0.0
            
        return next_action_value

    
    # 최대가 여러 개이면 랜덤으로 선택한다.
    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            
            elif value == max_value:
                max_index_list.append(index)
                
        return random.choice(max_index_list)




if __name__ == "__main__":
   
    agent = Agent()
    environment = Environment()
    
    agent.environment = environment
    environment.agent = agent
    
    environment.mainloop()
    