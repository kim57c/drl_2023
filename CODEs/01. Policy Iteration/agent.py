
import random
from environment import GridWorld, Enveronment

from environment import ROW, COLUMN, ACTIONS, TARGET


class Agent :
    
    def __init__(self, enveronment):
        
        # 환경 저장
        self.enveronment = enveronment
        
        # 감가율
        self.discount_factor = 0.9

        self.reset()
        
        
    # 가치함수와 정책 테이블을 리셋한다.
    def reset(self):
        
        # 가치함수를 2차원 리스트로 초기화
        self.value_table = [[0.0]*COLUMN for _ in range(ROW)]
        
        # 상,하, 좌, 우 동일한 확률로 정책을 초기화
        self.policy_table = [[[0.25, 0.25, 0.25, 0.25]]*COLUMN for _ in range(ROW)]
        
        # 성공 상태의 설정
        self.policy_table[2][2] = []
        

    # 정책 평가
    def policy_evaluation(self):

        # 다음 가치함수 초기화, 동기 백업을 위해 
        next_value_table = [[0.00]*COLUMN for _ in range(ROW)]

        # 모든 상태에 대해서 벨만 기대방정식을 계산
        for state in self.enveronment.get_all_states():
            
            value = 0.0
            
            # 성공적인 종료 상태의 가치 함수 = 0
            if state == [2, 2]:
                next_value_table[state[0]][state[1]] = value
                continue
            
            # 벨만 기대 방정식
            for action in ACTIONS:
                
                # 액션을 취한 다음 상태
                next_state = self.enveronment.state_after_action(state, action)
                
                # 액션에 대한 보상
                reward = self.enveronment.get_reward(state, action)
                
                # 다음 가치 함수
                next_value = self.get_value(next_state)
                
                # 업데이트 가치 함수 계산
                value += (self.get_policy(state)[action]*(reward+self.discount_factor*next_value))

            next_value_table[state[0]][state[1]] = round(value, 2)

        
        # 동기 백업
        self.value_table = next_value_table



    # 정책 개선
    def policy_improvement(self):
        
        next_policy = self.policy_table
        
        for state in self.enveronment.get_all_states():
            if state == TARGET:
                continue
            value = -99999
            max_index = []
            
            # 반환할 정책 초기화
            result = [0.0, 0.0, 0.0, 0.0]
            
            # 모든 행동에 대해서 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for index, action in enumerate(ACTIONS):
                
                # 환경에 액션을 주고 다음 상태를 얻어온다.
                next_state = self.enveronment.state_after_action(state, action)
                
                # 환경에 상태와 액션을 주고 보상을 얻어온다.
                reward = self.enveronment.get_reward(state, action)
                
                # 다음 상태의 가치함수 가져오기
                next_value = self.get_value(next_state)
                
                # 이 액션에 대한 기대값을 계산
                action_value = reward + self.discount_factor * next_value

                # 받을 보상이 최대인 행동의 index(최대가 복수라면 모두)를 추출
                if action_value == value:
                    max_index.append(index)
                    
                elif action_value > value:
                    value = action_value
                    max_index.clear()
                    max_index.append(index)

            # 행동의 확률 계산
            prob = 1/len(max_index)

            for index in max_index:
                result[index] = prob

            # 업데이트할 폴리시 리스트에 액션의 확률을 저장해 둔다.
            next_policy[state[0]][state[1]] = result

        # 동기 백업
        self.policy_table = next_policy


    # 학습이 종료되고 에이전트로 실행할 때 액션을 결정하는 함수
    def get_action(self, state):
        
        # 0 ~ 1 사이의 값을 무작위로 추출
        random_pick = random.randrange(100)/100

        # 현재 상태의 정책(방향에 대한 확률)을 가져오기
        policy = self.get_policy(state)
        
        policy_sum = 0.0
        # 정책에 담긴 행동 중에 무작위로 한 행동을 추출
        for index, value in enumerate(policy):
            policy_sum += value
            if random_pick < policy_sum:
                return index


    # 상태에 해당하는 정책 반환
    def get_policy(self, state):
        if state == TARGET:
            return 0.0
        return self.policy_table[state[0]][state[1]]


    # 상태에 해당하는 가치 함수의 값을 반환
    def get_value(self, state):
        # 소숫점 둘째 자리까지만 계산
        return round(self.value_table[state[0]][state[1]], 2)




if __name__ == "__main__":
    
    enveronment = Enveronment()
    agent = Agent(enveronment)
    
    grid_world = GridWorld(agent)
    grid_world.mainloop()