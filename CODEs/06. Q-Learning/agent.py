import numpy as np
import time

from environment import Environment
from environment import HEIGHT, WIDTH
ACTIONS = [0,1,2,3]


class Agent:
    
    def __init__(self):
        
        self.learning_rate = 0.05 # 학습률
        self.discount_factor = 0.9 # 할인율 
        self.epsilon = 0.1 # 입실론

        self.reset()
        
        
    def reset (self):
        # 큐(액션 가치) 테이블 초기화
        self.q_table = {}
        
    def get_state_actions ( self, state ):
        state = str(state)
        return self.q_table[state] if state in self.q_table else None
        
    def get_q_value (self, state, action ):
        actions = self.get_state_actions (state)
        if actions == None :
            return 0
        if str(action) in actions :
            return actions[str(action)] 
        return 0 
    
    
    def set_q_value (self, state, action, value):
        actions = self.get_state_actions (state)
        if actions == None :
            actions={}
            self.q_table[str(state)] = actions
        actions[str(action)] = value
    
    
    def episode (self, show=True):
        
        done = False 
        state = self.environment.reset()
        
        # 현재 상태에서 액션 구하기
        action = self.get_action(state)

        # 에피소드가 종료될 때까지 
        while not done:
            
            # 이동을 그리기 위해
            if show :
                self.environment.redraw()
                time.sleep(0.3)
            
            # 환경에 다음 스텝을 진행하고, 다음 상태, 보상, 종료여부를 반환 받는다.
            next_state, reward, done = self.environment.step(action)
            
            # 다음 상태 기준으로 다음 액션을 가져온다.
            next_action = self.get_action(next_state)
            
            #  Q 업데이트
            current_q = self.get_q_value ( state, action )
            next_actions = self.get_state_actions (next_state)
            next_q = max(next_actions.values()) if next_actions != None else 0
            
            # Q(S,A) = Q(S,A) + a*( R + r * max(Q(S',A')) - Q(S,A))
            new_q = current_q + self.learning_rate * \
                ( reward + self.discount_factor * next_q - current_q )
            
            self.set_q_value ( state, action, new_q )
            
            state = next_state 
            action = next_action 
            
            if show :
                self.environment.write_q_table ( self.q_table )
                self.environment.redraw()
        
        self.environment.write_q_table ( self.q_table )

        if show :
            self.environment.redraw()    
        
        
        
    # 랜덤값이 입실론보다 작으면 랜덤으로하고, 그렇지 않으면 최대 가치함수로 선택(입실론 그리디 정책) 
    def get_action(self, state):
        
        if np.random.rand() < self.epsilon:
            # 랜덤 액션
            action = np.random.choice(ACTIONS)
        else:
            # 큐테이블에서 그리디 액션
            state_actions = self.get_state_actions(state)
            action = self.arg_max(state_actions)
        return action
    
    
    # 최대가 다수개이면 랜덤으로 선택한다.
    @staticmethod
    def arg_max(actions):
        if actions == None :
            return np.random.choice(ACTIONS)
            
        max_index_list = []
        max_value = -np.inf
        
        for key, value in actions.items():
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(int(key))
            elif value == max_value:
                max_index_list.append(int(key))
        
        return np.random.choice(max_index_list)




if __name__ == "__main__":
   
    agent = Agent()
    environment = Environment()
    
    agent.environment = environment
    environment.agent = agent
    
    environment.mainloop()
    