
#######################################################################

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#######################################################################

# 하이퍼 파라메터들
learning_rate = 0.0005 # 학습률
gamma         = 0.98 # 할인률
lmbda         = 0.95 # 람다
eps_clip      = 0.1 # 클립
K_epoch       = 3 # 하나의 배치 당 훈련 에포크 수
T_horizon     = 20 # 배치 단위 

#######################################################################

class Ppo(nn.Module):
    
    def __init__(self):
        super(Ppo, self).__init__()
        self.data = []

        self.fc1   = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
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
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
        
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch): # K_epoch 만큼 수행
            
            # A(t)=Q(t)-V(t), Q(t)=r(t)+감마(할인율)*V(st+1)
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = lmbda * advantage + delta_t[0]
                # advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.pi(s, softmax_dim=1)
            # pi에서 a 위치에 있는 값들을 반환 
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage 
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

#######################################################################

def main():
    
    env = gym.make('CartPole-v1')
    model = Ppo()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        
        s, _ = env.reset()
        done = False
        
        while not done: # 에피소드의 종료까지 진행 
            
            for t in range(T_horizon): # T_horizon 이거나, 종료까지 진행 
                
                # 현재 정책으로 다음 액션을 설정한다.
                # prob는 2차원(왼쪽, 오른쪽 확률) 리스트 텐서
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                
                s_prime, r, done, info, _ = env.step(a)

                # 버퍼에 트라젝토리 구성
                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
                s = s_prime

                # if score > 5000  :
                #     print(score, end=' ' )
                
                score += r
                if done:
                    break
                
            # 현재 구성된 버퍼로 훈련 수행
            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print(f"Episode :{n_epi}, avg score : {score/print_interval:.1f}")
            score = 0.0

    env.close()


#######################################################################


if __name__ == '__main__':
    main()