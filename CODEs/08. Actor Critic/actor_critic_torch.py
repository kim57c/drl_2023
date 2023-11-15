
######################################################################

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

######################################################################

learning_rate = 0.0002 # 학습율
gamma = 0.98 # 할인율
n_rollout = 10 # 배치크기

######################################################################

class ActorCritic(nn.Module):
    
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4,256)
        
        # 액션의 확률을 위한 출력 
        self.fc_pi = nn.Linear(256,2)
        
        # 가치함수를 위한 출력
        self.fc_v = nn.Linear(256,1)
        
        # 신경망의 최적화기 정의
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v 
    
    def put_data(self, transition):
        self.data.append(transition)
    
    # 배치 작업을 위해 데이터들을 텐서화 한다.
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])
            
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), \
                                                               torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), \
                                                               torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch


    def train_net(self):
    
        s, a, r, s_prime, done = self.make_batch()
        
        # TD 타겟 = 보상 + 할인율 * 다음가치함수
        td_target = r + gamma * self.v(s_prime) * done
        
        # TD에러(델타) = TD 타겟 - 현재가치함수
        delta = td_target - self.v(s)
        
        pi = self.pi(s, softmax_dim=1)
        
        # pi.shape = 10,2, a.shape = 10,1
        # pi에서 a 위치에 있는 값들을 반환 
        pi_a = pi.gather(1, a)
        
        # 두개의 함수를 하나의 로스로 모두 학습하기 위해 앞쪽은 파이(Policy)로스, 뒤쪽은 V(Value Function)로스
        loss = -torch.log(pi_a) * delta.detach() + \
            F.smooth_l1_loss ( self.v(s), td_target.detach() )
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
        
######################################################################
        
def main():
    
    # env = gym.make('CartPole-v1', render_mode="human")
    env = gym.make('CartPole-v1')
    
    model = ActorCritic()
    print_interval = 20 
    score = 0.0
    
    for n_epi in range(1000):
        done = False
        s, _ = env.reset()
        
        while not done :
            for _ in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, done, truncated, _ = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                
                s = s_prime
                score += r
                
                if done or truncated:
                    break
                
            model.train_net()
            
        if n_epi%print_interval==0 and n_epi!=0:
            print(f"Episode :{n_epi}, Avg score : { score/print_interval:.1f}")
            score = 0.0
    env.close()
    
    
if __name__ == '__main__':
    main()
    

######################################################################