import numpy as np
import random
import time

from environment import Environment
from environment import HEIGHT, WIDTH
ACTIONS = [0,1,2,3]


class Agent:
    
    def __init__(self):
        
        self.learning_rate = 0.1 # 학습률
        self.discount_factor = 0.9 # 할인율 
        self.epsilon = 0.1 # 입실론

        self.reset()
        
        
    def reset (self):
         # 가치함수 테이블 초기화
        self.value_table = {}
        
        
    def episode (self, show=True):
        
        done = False
        
        state = self.environment.reset()
        # action = self.get_action(state)
        
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
            
            # 방문마다 업데이트 
            value = self.value_table[str(state)] if str(state) in self.value_table else 0.0
            next_value = self.value_table[str(next_state)] if str(next_state) in self.value_table else 0.0
            
            #  가장 단순한 TD 학습 알고리즘 : TD(0)
            value = value + self.learning_rate * ( reward + self.discount_factor * next_value - value )
            self.value_table[str(state)] = value 

            if show :
                self.environment.write_value_table ( self.value_table )
                self.environment.redraw()

            state = next_state 
            
        # 종료상태의 가치함수 업데이트
        value = self.value_table[str(next_state)] if str(next_state) in self.value_table else 0
        value = value + self.learning_rate*reward
        self.value_table[str(next_state)] = value
        
        self.environment.write_value_table ( self.value_table )

        if show :
            self.environment.redraw()    
        
        
        
    # 랜덤값이 입실론보다 작으면 랜덤으로하고, 그렇지 않으면 최대 가치함수로 선택(입실론 그리디 정책) 
    def get_action(self, state):
        
        p = np.random.random()
        if np.random.random() > self.epsilon:
            action_values = self.possible_next_action_values(state)
            action = self.arg_max(action_values)
        else:
            action = np.random.choice(ACTIONS)

        return int(str(action))



    # 가능한 다음 상태 반환
    def possible_next_action_values(self, state):
        col, row = state
        next_action_value = [0.0] * 4

        if row == 0 :
            next_action_value[0] = -9999.0
        else :
            key = str([col,row-1])
            next_action_value[0] = self.value_table[key] if key in self.value_table else 0.0
        
        if row == HEIGHT-1 :
            next_action_value[1] = -9999.0
        else :
            key = str([col,row+1]) 
            next_action_value[1] = self.value_table[key] if key in self.value_table else 0.0
            
        if col == 0 :
            next_action_value[2] = -9999.0
        else :
            key = str([col-1,row])
            next_action_value[2] = self.value_table[key] if key in self.value_table else 0.0
            
        if col == WIDTH-1 :
            next_action_value[3] = -9999.0
        else :
            key = str([col+1,row])
            next_action_value[3] = self.value_table[key] if key in self.value_table else 0.0
            
        return next_action_value

    
    # 최대가 다수개이면 랜덤으로 선택한다.
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
    