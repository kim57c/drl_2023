# -*- coding: utf-8 -*-
from environment import GridWorld, Enveronment

from environment import ROW, COLUMN, ACTIONS, TARGET


class Agent:
    
    def __init__(self, enveronment):
        self.enveronment = enveronment        
        self.discount_factor = 0.9

        self.reset()
        
        
    def reset(self):
        
        # 가치 테이블 초기화 
        self.value_table = [[0.0]*COLUMN for _ in range(ROW)]
         
    
    # 현재의 가치 테이블로 다음 가치 테이블을 계산
    def value_iteration(self):

        next_value_table = [[0.0]*COLUMN for _ in range(ROW)]
        
        for state in self.enveronment.get_all_states():
            
            if state == TARGET: #[2, 2]:  # 성공적인 종료 상태의 가치 함수 = 0
                next_value_table[state[0]][state[1]] = 0.0
                continue
            
            value_list = []

            # 액션별로 업데이트 가치 함수 계산 
            for action in ACTIONS :
                
                # 액션을 수행한 다음 상태
                next_state = self.enveronment.state_after_action(state, action)
                
                # 액션에 대한 보상
                reward = self.enveronment.get_reward(state, action)
                
                # 다음 가치 함수
                next_value = self.get_value(next_state)
                
                # 이번 얙션의 업데이트 가치 함수 계산 ( 보상 + 할일계수 * 다음 가치 함수 )
                value_list.append((reward + self.discount_factor * next_value))
                
            # 벨만 최적 방정식으로 최대 가치함수를 저장
            next_value_table[state[0]][state[1]] = round(max(value_list), 2)
            
        self.value_table = next_value_table


    # 학습이 종료되고 에이전트로 실행할 때 액션을 결정하는 함수
    def get_action(self, state):
        
        max_value = -99999
        if state == TARGET: #[2, 2]:
            return []

        # 최대가 동일한 가치함수 경우 액션이 여러개
        
        action_list = []
        for action in ACTIONS :

            # 액션을 수행한 다음 상태
            next_state = self.enveronment.state_after_action(state, action)
            
            # 액션에 대한 보상
            reward = self.enveronment.get_reward(state, action)
            
            # 다음 가치 함수
            next_value = self.get_value(next_state)
            
            # 이번 얙션의 업데이트 가치 함수 계산 ( 보상 + 할일계수 * 다음 가치 함수 )
            value = (reward + self.discount_factor * next_value)

            if value > max_value:
                action_list.clear()
                action_list.append(action)
                max_value = value
            elif value == max_value:
                action_list.append(action)

        return action_list

    
    # 상태에 해당하는 가치 함수의 값을 반환
    def get_value(self, state):
        return round(self.value_table[state[0]][state[1]], 2)



if __name__ == "__main__":
    
    enveronment = Enveronment()
    agent = Agent(enveronment)
    
    grid_world = GridWorld(agent, enveronment)
    grid_world.mainloop()
