
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

import tensorflow as tf

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

class DQN(tf.keras.Model):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = tf.keras.layers.Dense(n_observations)
        self.layer2 = tf.keras.layers.Dense(128, activation='relu')
        self.layer3 = tf.keras.layers.Dense(128, activation='relu')
        self.layer4 = tf.keras.layers.Dense(n_actions)
        
    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

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
target_net.set_weights(policy_net.get_weights()) 

# 목적(비용) 함수
criterion = tf.keras.losses.Huber()

# 옵티마이저 정의 
optimizer = tf.keras.optimizers.AdamW(learning_rate=LR, amsgrad=True)

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
        
        # 현재의 정책 네트워크로 액션값을 구한다.
        actions = policy_net(state, training=False)

        # argmax 함수는 가장 큰 값의 인덱스를 리스트로 반환한다.
        # expand_dims함수는 텐서의 차원(shape)을 확장 해준다.
        return tf.expand_dims(tf.argmax(actions,1),0)

    else:
        return tf.convert_to_tensor([[env.action_space.sample()]], dtype=tf.int64)


######################################################################
# plot 그리기

episode_durations = []

def plot_durations(show_result=False):
    plt.figure('Tensorflow')
    durations_t = tf.convert_to_tensor(episode_durations,dtype=tf.float32)
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
    res = [i for i in range(len(batch.next_state)) if batch.next_state[i] != None]

    # m = map(lambda s: s is not None, batch.next_state)
    # t = tuple(m)
    non_final_mask = tf.convert_to_tensor(res,dtype=tf.int32)
    # non_final_mask = tf.expand_dims(tuple(map(lambda s: s is not None, batch.next_state),dtype=tf.bool),0)
    
    # non_final_mask.shape=128
    
    # batch.next_state = 128, 1, 4
    nfns = [s for s in batch.next_state if s is not None]
    # nfns는 텐서들의 리스트 124, 1, 4
    non_final_next_states = tf.concat(nfns,0)
    # non_final_next_states.shape=124,4
    
    # batch.state는 튜플 shape(batch.state)=128,1,4
    state_batch = tf.concat(batch.state,0)
    # state_batch.shape = 128,4
    
    # batch.action은 튜플 shape(batch.action)=128,1,1
    action_batch = tf.concat(batch.action,0)
    # action_batch.shape = 128,1
    
    # batch.reward는 튜플 shape(batch.reward)=128,1
    reward_batch = tf.concat(batch.reward,0)
    # reward_batch.reward = 128
    
    # 다음 상태를 위한 Q(s_{t+1}) 계산
    next_q_values = tf.zeros(BATCH_SIZE)
    
    next_state_values = target_net(non_final_next_states,training=False)
    # reduce_max는 축에 해당하는 최대값을 반환
    next_state_values = tf.reduce_max(next_state_values, axis=1)
    non_final_mask = tf.expand_dims(non_final_mask,1)
    next_q_values = tf.tensor_scatter_nd_update(next_q_values, non_final_mask, next_state_values)
    
    # 새로운 Q-벨류 = 보상 + 할인율 * 다음 Q-벨류
    expected_q_values = reward_batch + GAMMA * next_q_values
    expected_q_values = tf.expand_dims(expected_q_values,1)
    optimize ( state_batch, action_batch, expected_q_values )


@tf.function
def optimize(state_batch, action_batch,expected_q_values):
    with tf.GradientTape() as tape:
        # 현재 Q-벨류 Q(s_t, a) 계산
        state_action_values = policy_net(state_batch, training=True)
        # gather는 state_action_values의 1차원에서 action_batch(index)에 해당하는 값들을 반환
        q_values = tf.gather(state_action_values, action_batch, batch_dims=1)
        
        loss = criterion(q_values, expected_q_values)

    gradients = tape.gradient(loss, policy_net.trainable_variables)
    gradients = [(tf.clip_by_value(grad, -100., 100.)) for grad in gradients]
    optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))


    

######################################################################
# 학습
num_episodes = 500

for i_episode in range(num_episodes):
    
    # 환경과 상태 초기화
    state, info = env.reset()
    # state 크기가 4(상태)인 리스트 
    # state = tf.Tensor(state, dtype=tf.float32).unsqueeze(0)
    state = tf.expand_dims(tf.convert_to_tensor(state, dtype=tf.float32),0)
    # state.shape = 1,4
    
    for t in count():
        
        action = select_action ( state )
        # action.shape=1,1 action.view(-1).numpy()[0]
        
        observation, reward, terminated, truncated, _ = env.step ( action.numpy()[0][0] )
        # observation는 1차원 배열 크기는 4
        reward = tf.convert_to_tensor([reward])
        
        # terminated는 넘어진 상태(12도 이상), truncated는 에피소드가 500이상 된 경우 
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32),0)
            # next_state.shape = 1,4
            
        # 메모리에 트랜지션을 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # 최적화 한단계 수행
        optimize_model()

        # 목표 네트워크를 정책 네트워크로 소프트 업데이트 한다.
        # 타겟네트워크 = TAU * 정책네트워크 + (1−TAU) * 타겟네트워크
        
        weights = []
        targets = policy_net.weights
        for i, weight in enumerate(target_net.weights):
            weights.append(targets[i] * TAU + weight*(1-TAU))
        target_net.set_weights(weights)
        
        if done:
            episode_durations.append(t+1)
            plot_durations()
            break


print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()


