
######################################################################
"""
카트폴 게임 
https://gymnasium.farama.org/environments/classic_control/cart_pole/

- 막대가 똑바로 서 있도록, 카트를 왼쪽이나 오른쪽으로 움직이는 두 가지 동작 중 하나를 선택해야 함
- 매 타임스텝마다 보상은 +1
- 폴의 각도가 ±12도 이상이거나, 카트의 위치가 중심에서 ±2.4 유닛 이상 멀어지면 Termination
- 에피소드가 500이상 되면 Truncation
- 입력:상태(카트 위치, 카트 속도, 폴 각도, 폴 각속도) 4개, 출력:액션 상태 함수(왼쪽,오른쪽) 2개
"""
######################################################################

import gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch

env = gym.make("CartPole-v1")

# interactive-on : 함수 호출뒤에 바로 그래픽을 그려줌 
plt.ion()

######################################################################
# 리플라이 메모리(Replay Memory)

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


######################################################################
# 모델 네트워크 정의

class DQN(torch.nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)


######################################################################
# 하이퍼 파라미터

BATCH_SIZE = 128 # 리플레이 버퍼에서 샘플링된 트랜지션의 수
GAMMA = 0.99 # 할인 계수
EPS_START = 0.9 # EPS_START는 엡실론의 시작 값
EPS_END = 0.05  # EPS_END는 엡실론의 최종 값
EPS_DECAY = 1000 # 엡실론의 지수 감쇠(exponential decay) 속도 제어, 높을수록 감쇠 속도가 느림
TAU = 0.005 # 목표 네트워크의 업데이트 속도
LR = 1e-4 # 옵티마이저의 학습율

######################################################################
# 모델

# gym 액션 공간에서 액션의 전체 수를 얻는다.
n_actions = env.action_space.n

# 환경을 리셋하여 초기 상태와 정보를 앋는다.
state, info = env.reset()
n_observations = len(state)

# 가치함수를 근사하는 네트워크 
policy_net = DQN(n_observations, n_actions)

# 고정된 Q 타겟으로 사용할 네트워크 
target_net = DQN(n_observations, n_actions)

# 두 네트워크를 동일한 값으로 로드 함
target_net.load_state_dict(policy_net.state_dict())

# 목적함수 정의
criterion = torch.nn.SmoothL1Loss()
    
# 옵티마이저 정의 
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# 리플라이 메모리
memory = ReplayMemory(10000)


######################################################################
# 입실론 디케이로 액션 선택

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END+(EPS_START-EPS_END)*math.exp(-1.*steps_done/EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            
            # 현재의 정책 네트워크로 액션값을 구한다.
            actions = policy_net(state)

            # max 함수는 가장 큰 값[0]과 그 값의 인덱스[1]를 리스트로 반환한다.
            # view함수는 텐서의 차원(shape)을 변경(1->1,1로) 해준다.
            return actions.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], dtype=torch.long)



######################################################################
# plot 그리기

episode_durations = []

def plot_durations(show_result=False):
    plt.figure('Pytorch')
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    
    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    

######################################################################
# 모델 학습

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    # BATCH_SIZE 크기만큼 샘플링
    transitions = memory.sample(BATCH_SIZE)
    
    # 상태 4가지를 각각 리스트로 전환한다. 
    # (s11,s12,s13,s14),(s21,s22,s23,s24)...... -> (s11,s21,s31,s41...),(s12..),(),()
    # zip 함수는 여러개의 리스트를 4개의 변수를 
    batch = Transition(*zip(*transitions))

    # shape(batch.next_state)=128,1,4
    m = map(lambda s: s is not None, batch.next_state)
    t = tuple(m)
    non_final_mask = torch.tensor(t, dtype=torch.bool)
    # non_final_mask.shape=128
    
    # torch.cat는 텐서의 차원을 concatenate
    # batch.next_state = 128, 1, 4
    nfns = [s for s in batch.next_state if s is not None]
    # nfns는 텐서들의 리스트 124, 1, 4
    non_final_next_states = torch.cat(nfns)
    # non_final_next_states.shape=124,4
    
    # batch.state는 튜플 shape(batch.state)=128,1,4
    state_batch = torch.cat(batch.state)
    # state_batch.shape = 128,4
    
    # batch.action은 튜플 shape(batch.action)=128,1,1
    action_batch = torch.cat(batch.action)
    # action_batch.shape = 128,1
    
    # batch.reward는 튜플 shape(batch.reward)=128,1
    reward_batch = torch.cat(batch.reward)
    # reward_batch.reward = 128
    
    # Q(s_t, a) 계산, 정책 네트워크로 현재 상태에 대한 결과들을 가져옴
    state_action_values = policy_net(state_batch)
    
    # gather는 sav의 1차원에서 action_batch(index)에 해당하는 값으로 텐서를 반들어 반환
    # sav = [[1,2],[2,3],[3,4]...] action_batch= [[1],[0],[1] ...]
    
    # 현재 Q-벨류(액션 가치 함수)
    q_values = state_action_values.gather(1, action_batch)
    # q_values.shape = 128,1
    
    # 다음 상태를 위한 Q(s_{t+1}) 계산
    next_q_values = torch.zeros(BATCH_SIZE)
    
    with torch.no_grad():
        nsv = target_net(non_final_next_states)
        #  max 함수는 가장 큰 값[0]과 그 인텍스[1]를 튜플로 반환 
        next_q_values[non_final_mask] = nsv.max(1)[0]
    
    # 새로운 Q-벨류 = 보상 + 할인율 * 다음 Q-벨류
    expected_q_values = reward_batch+ GAMMA * next_q_values

    optimize(q_values, expected_q_values)
    
    
def optimize ( q_values, expected_q_values ):
    
    # squeeze는 차원이 1인 차원을 제거하고, unsqueeze는 지정한 차원자리에 1인 빈차원을 채워 차원을 확장
    loss = criterion(q_values, expected_q_values.unsqueeze(1))
    
    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 클리핑 : 너무 크거나 작은 gradient의 값을 제한 -100이하 이거나 100 이상
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

######################################################################
# 학습
num_episodes = 500

for i_episode in range(num_episodes):
    
    # 환경과 상태 초기화
    state, info = env.reset()
    # state 크기가 4(상태)인 리스트 
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    # state.shape = 1,4
    
    for t in count():
        
        action = select_action ( state )
        # action.shape=1,1 
        observation, reward, terminated, truncated, _ = env.step ( action.item() )
        # observation는 1차원 배열 크기는 4
        reward = torch.tensor([reward])
        
        # terminated는 거의 넘어진 상태(12도 이하), truncated는 에피소드가 500이상 된 경우 
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            # next_state.shape = 1,4
            
        # 메모리에 트랜지션을 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # 최적화 한단계 수행
        optimize_model()

        # 목표 네트워크를 정책 네트워크로 소프트 업데이트 한다.
        # 타겟네트워크 = TAU * 정책네트워크 + (1−TAU) * 타겟네트워크
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        
        target_net.load_state_dict ( target_net_state_dict )

        if done:
            episode_durations.append(t+1)
            plot_durations()
            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
