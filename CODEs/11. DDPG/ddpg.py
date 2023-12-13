
'''
Pendulum (진자) 게임

- CartPole이 이산액션공간 인것에 비해 Pendulum은 연속액션공간이다.
- https://www.gymlibrary.dev/environments/classic_control/pendulum/

- 역진자 스윙업 문제는 제어 이론의 고전적인 문제에 기초한다.
- 이 시스템은 한쪽 끝이 고정된 지점에 부착되고 다른 쪽 끝은 자유로워진 진자로 구성된다. 
- 진자는 무작위 위치에서 시작한다.
- 목표는 진자에 토크를 가해서, 무게 중심이 고정점의 바로 위에 있는 수직 위치로 가게 하는 것이다.

- 액션은 토크이며 -2 ~ 2 사이의 값이다.
- Observation Space(상태가 됨)은 진자 끝의 X,Y 좌표와 각속도로 나타낸 3차원 
    X = cos(theta) -1.0 ~ 1.0
    Y = sin(angle) -1.0 ~ 1.0
    각속도(Angular Velocity) = -8.0 ~ 8.0
- 보상 r = -(theta^2 + 0.1 * theta_dt^2 + 0.001 * torque^2)

'''


import gym
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#####################################################
# 하이퍼 파라메터들

lr_mu        = 0.0005 # mu의 학습률
lr_q         = 0.001 # q의 학습률
gamma        = 0.99 # 할인률
batch_size   = 32 # 배치 크기 
buffer_limit = 50000 # 버퍼의 최대 크기 
tau          = 0.005 # 소프트 업데이트를 위한 타우 값 

#####################################################
# 리플라이 버퍼 

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)


#####################################################
# Mu와 Q네트워크 정의 

class MuNet(nn.Module):
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 
        return mu
    
    
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(3, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q


#####################################################
# 온슈타인-울렌벡 노이즈 - 평균으로 회귀

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
      
      
#####################################################
# 학습 
def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    s,a,r,s_prime,done_mask = memory.sample(batch_size)
    
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() 
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()


# 소프트 업데이트
def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


#####################################################
# 메인 함수 
def main():
    env = gym.make('Pendulum-v1', max_episode_steps=200, autoreset=True)
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20

    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    for n_epi in range(10000):
        s, _ = env.reset()
        done = False

        count = 0
        while count < 200 and not done:
            a = mu(torch.from_numpy(s).float()) 
            a = a.item() + ou_noise()[0]
            s_prime, r, done, truncated, info = env.step([a])
            memory.put((s,a,r/100.0,s_prime,done))
            score +=r
            s = s_prime
            count += 1
        
        if memory.size()>2000:
            for i in range(10):
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q, q_target)
        
        if n_epi%print_interval==0 and n_epi!=0:
            print(f"에피소드 :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()