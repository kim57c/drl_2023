
######################################################################

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from btc_krw_env import BtcEnv

######################################################################

learning_rate = 0.0002 # 학습율
gamma = 0.98 # 할인율
n_rollout = 10 # 배치크기

######################################################################

clear_console = lambda: os.system('cls' if os.name in ('nt', 'dos') else 'clear')


class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'


class ActorCritic(nn.Module):
    
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        input_size = 24
        
        nets = [4, 2]
        
        s = input_size
        self.layers = nn.ModuleList ()
        
        for net in nets:
            size = round(input_size*net)
            self.layers.append(nn.Linear(s, size))
            s = size
        
        # 아웃풋  설정 
        self.fc_pi = nn.Linear(s, 2)
        self.fc_v  = nn.Linear(s, 1) 
        
        # 신경망의 최적화기 정의
        self.optimizer = optim.NAdam(self.parameters(), lr=learning_rate)
        
        
    def pi(self, x, softmax_dim=0):
        
        for _, layer in enumerate(self.layers):
            x = F.normalize(x, dim=len(x.shape)-1)
            x = F.relu(layer(x))
        
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
            
        return prob
    
    
    def v(self, x):
        
        for _, layer in enumerate(self.layers):
            x = F.normalize(x, dim=len(x.shape)-1)
            x = F.relu(layer(x))

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
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss ( self.v(s), td_target.detach() )
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        
        
######################################################################

def print_log(n_epi, env):
    
    cash, btc = env.assets()
    assets = cash+btc
    
    print(f"에피소드:[{style.BLUE}{n_epi}{style.RESET}],", end=" ")
    print(f"시작자산:[{style.CYAN}{format(round(env.seed_money),',')}{style.RESET}],", end=" ")
    print(f"현재자산:[{style.YELLOW}{format(round(assets),',')}{style.RESET}],", end=" ")
    
    rate = assets/10000*100
    rate = rate-100
    
    if rate>0 :
        print(f'{style.GREEN}(▲+{rate:.1f}%){style.RESET}', end='')
    else :
        print(f'{style.RED}(▼{rate:.1f}%){style.RESET}', end='')
        
    print(f'')
        
    
        
def main():
    
    clear_console()
    
    env = BtcEnv()
    
    model = ActorCritic()
    
    
    for n_epi in range(100):
        done = False
        s = env.reset()
        
        while not done :
            for _ in range(n_rollout):
                
                prob = model.pi(torch.tensor(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, done = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                
                s = s_prime
                
                if done :
                    break
                
            model.train_net()
        
        print_log (n_epi, env)
        
    
    
if __name__ == '__main__':
    main()
    

######################################################################