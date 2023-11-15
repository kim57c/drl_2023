
######################################################################

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np

######################################################################

# 하이퍼 파라메터들
n_train_processes = 3 # 멀티 프로세스 수
learning_rate = 0.0002 # 학습율
update_interval = 5 # 업데이트 주기
gamma = 0.98 # 할인 계수
max_train_steps = 60000 # 최대 훈련 횟수 
PRINT_INTERVAL = update_interval * 100 # 화면 로그 프린트 간격

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


def worker(worker_end):
    env = gym.make('CartPole-v1')

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, t, _ = env.step(data)
            if done:
                ob, _ = env.reset()
            worker_end.send((ob, [reward], [done]))
        elif cmd == 'reset':
            ob, _ = env.reset()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        else:
            raise NotImplementedError

######################################################################

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = list()

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_end in worker_ends:
            p = mp.Process(target=worker, args=(worker_end,))
            p.daemon = True
            p.start()
            self.workers.append(p)

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

######################################################################

def test(step_idx, model):
    env = gym.make('CartPole-v1')
    score = 0.0
    done = False
    num_test = 10

    for _ in range(num_test):
        s, _ = env.reset()
        while not done:
            prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().numpy()
            s_prime, r, done, info, _ = env.step(a)
            s = s_prime
            score += r
        done = False
    print(f"Step :{step_idx}, Avg score : {score/num_test:.1f}")

    env.close()


def compute_target(v_final, r_lst, mask_lst):
    G = v_final.reshape(-1)
    td_target = list()

    for r, mask in zip(r_lst[::-1], mask_lst[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()


######################################################################


if __name__ == '__main__':
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step_idx = 0
    s = envs.reset()
    while step_idx < max_train_steps:
        s_lst, a_lst, r_lst, mask_lst = list(), list(), list(), list()
        for _ in range(update_interval):
            prob = model.pi(torch.from_numpy(s).float())
            a = Categorical(prob).sample().numpy()
            s_prime, r, done  = envs.step(a)

            s_lst.append(s)
            a_lst.append(a)
            r_lst.append(r/100.0)
            mask_lst.append(1 - done)

            s = s_prime
            step_idx += 1

        s_final = torch.from_numpy(s_prime).float()
        v_final = model.v(s_final).detach().clone().numpy()
        
        # TD 타겟 구하기
        td_target = compute_target(v_final, r_lst, mask_lst)

        td_target_vec = td_target.reshape(3,-1)
        s_vec = torch.tensor(s_lst).float().reshape(-1, 4)  
        a_vec = torch.tensor(a_lst).reshape(-1).unsqueeze(1)
        
        # 어드밴티치 구하기
        advantage = td_target_vec - model.v(s_vec).reshape(-1)

        pi = model.pi(s_vec)
        # pi에서 a 위치에 있는 값들을 반환 
        pi_a = pi.gather(1, a_vec).reshape(-1)
        
        # 두개의 함수를 하나의 로스로 모두 학습하기 위해 앞쪽은 파이(Policy)로스, 뒤쪽은 V(Value Function)로스
        loss = -(torch.log(pi_a) * advantage.detach()).mean() +\
            F.smooth_l1_loss(model.v(s_vec).reshape(-1), td_target_vec)
        
        # 기울기를 0으로 초기화
        optimizer.zero_grad()
        
        # 역전파 수행으로 모델의 기울기를 계산 
        loss.backward()
        
         # 각 파라메터를 조정
        optimizer.step()

        if step_idx % PRINT_INTERVAL == 0:
            test(step_idx, model)

    envs.close()