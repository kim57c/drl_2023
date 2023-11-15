
######################################################################

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import time
import numpy as np


######################################################################

# 하이퍼 파라메터들
n_train_processes = 3 # 멀티 프로세스 수
learning_rate = 0.00002 # 0.0002 # 학습율
update_interval = 5 # 업데이트 주기
gamma = 0.98 # 할인 계수
max_test_ep = 1000 # 검증 에피소드 횟수

######################################################################

class ActorCritic(nn.Module):
    
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=len(x.shape)-1)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

######################################################################

def train(global_model,rank,v):
    local_model = ActorCritic()
    
    # 글로벌 모델의 파라메터로 로컬 모델을 로드 
    local_model.load_state_dict(global_model.state_dict())

    # 옵티마이저는 글로벌 모델의 파라메터를 설정한다.
    optimizer = optim.Adam(global_model.parameters(), lr=learning_rate)

    env = gym.make('CartPole-v1')
    
    while v.value == 0:
        done = False
        s, _ = env.reset()
        while not done:
            s_lst, a_lst, r_lst = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, truncated, _ = env.step(a)

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r/100.0)

                s = s_prime
                if done:
                    break
            
            # TD 타겟을 리스트로 구하기
            s_final = torch.tensor(np.array(s_prime), dtype=torch.float)
            R = 0.0 if done else local_model.v(s_final).item()
            td_target_lst = []
            for reward in r_lst[::-1]:
                R = gamma * R + reward
                td_target_lst.append([R])
            td_target_lst.reverse()

            # 배치로 수행하기 위해 배치 리스트 구성
            s_batch, a_batch, td_target = torch.tensor(np.array(s_lst), dtype=torch.float),\
                torch.tensor(np.array(a_lst)), \
                torch.tensor(np.array(td_target_lst))
            
            # 어드밴티치 구하기
            advantage = td_target - local_model.v(s_batch)

            pi = local_model.pi(s_batch)
            # pi에서 a 위치에 있는 값들을 반환 
            pi_a = pi.gather(1, a_batch)
            
            # 두개의 함수를 하나의 로스로 모두 학습하기 위해 앞쪽은 파이(Policy)로스, 뒤쪽은 V(Value Function)로스
            loss = -torch.log(pi_a) * advantage.detach() + \
                F.smooth_l1_loss(local_model.v(s_batch), td_target.detach())

            # 기울기를 0으로 초기화
            optimizer.zero_grad()
            
            # 역전파 수행으로 로컬 모델의 기울기를 계산 
            loss.mean().backward()
            
            # 로컬 모델의 기울기들을 글로벌 모델의 기울기로 셋
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
                
            # 각 파라메터를 조정
            optimizer.step()
            
            # 다시 글로벌 모델의 파라메터로 로컬 모델을 로드 
            local_model.load_state_dict(global_model.state_dict())
            
    env.close()


######################################################################

def test(global_model,v):
    env = gym.make('CartPole-v1')
    score = 0.0
    print_interval = 20

    for n_epi in range(max_test_ep+1):
        done = False
        s, _ = env.reset()
        while not done:
            prob = global_model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().item()
            s_prime, r, done, truncated, _ = env.step(a)
            s = s_prime
            score += r

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"Episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0
            time.sleep(0.5)
    env.close()
    v.value = 1


######################################################################

if __name__ == '__main__':
    
    global_model = ActorCritic()
    global_model.share_memory()

    v = mp.Value('i',0)
    processes = []
    for rank in range(n_train_processes + 1):  # 검증을 위해 하나 더 실행 시킨다.
        if rank == 0:
            p = mp.Process(target=test, args=(global_model,v,))
        else:
            p = mp.Process(target=train, args=(global_model,rank,v,))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()