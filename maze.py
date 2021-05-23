import sys
import numpy as np
import math
import random
import time as tm
import gym
import gym_maze
env = gym.make("maze-random-10x10-v0")
MAZE_SIZE = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
NUM_BUCKETS = MAZE_SIZE  # one bucket per grid
MIN_EXPLORE_RATE = 0.001
MIN_LEARNING_RATE = 0.2
DECAY_FACTOR = np.prod(MAZE_SIZE, dtype=float) / 10.0
NUM_ACTIONS = env.action_space.n
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))

#Learning parameters
NUM_EPISODES=500000
MAX_T=1000
SOLVED_T=100
STREAK_TO_END=100
Q=np.zeros(NUM_BUCKETS + (NUM_ACTIONS,), dtype=float)


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = int(np.argmax(Q[state]))
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.8, 1.0 - math.log10((t+1)/DECAY_FACTOR)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


def train():
    learning_rate=get_learning_rate(0)
    explore_rate=get_explore_rate(0)
    discount_factor=0.99

    num_streaks=0
    env.render()

    for episode in range(NUM_EPISODES):
        obv=env.reset()
        s=state_to_bucket(obv)
        total_reward=0

        for t in range(MAX_T):
            a=select_action(s,explore_rate)
            obv,r1,d,_=env.step(a)
            s1=state_to_bucket(obv)
            total_reward+=r1

            best_q=np.amax(Q[s1])
            Q[s+(a,)]+=learning_rate*(r1+discount_factor*(best_q)-Q[s+(a,)])
            s=s1

            env.render()

            if env.is_game_over():
                sys.exit()

            if(d):
                print("Episode %d finished after %f time steps with total reward = %f (streak %d)."
                      % (episode, t, total_reward, num_streaks))

                if t <= SOLVED_T:
                    num_streaks += 1
                else:
                    num_streaks = 0
                break

            elif t >= MAX_T - 1:
                print("Episode %d timed out at %d with total reward = %f."
                      % (episode, t, total_reward))

            if num_streaks > STREAK_TO_END:
                break

            explore_rate = get_explore_rate(episode)
            learning_rate = get_learning_rate(episode)

        if num_streaks > STREAK_TO_END:
            break

def simulate():
    obv=env.reset()
    s=state_to_bucket(obv)
    d=False
    reward=0
    time=0
    env.render()
    tm.sleep(2)
    while(not d):
        action = int(np.argmax(Q[s]))
        obv,r1,d,_=env.step(action)
        env.render()
        tm.sleep(2)
        s1=state_to_bucket(obv)
        s=s1
        reward+=r1
        time+=1
    print("Simulation ended at time %d with total reward = %f."
          % (time, reward))


train()
while(True):
    d=input("Simulate?(y/n)")
    if(d=='y'):
        simulate()
    else:
        break
