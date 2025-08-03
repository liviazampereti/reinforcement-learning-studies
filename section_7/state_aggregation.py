import random
import gym
import gym.spaces
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Maze import Maze
from Maze import plot_policy, plot_tabular_cost_to_go, test_agent, plot_stats, seed_everything

import gym
import matplotlib.pyplot as plt

# Crie o ambiente e defina a semente

env = gym.make('MountainCar-v0')
seed_everything(env)

env.reset()

frame = env.render(mode='rgb_array')

plt.imshow(frame)
plt.show()

class StateAggregationEnv(gym.ObservationWrapper):

    def __init__(self, env, bins, low, high):
        # low = [-1.2, -0.07], high = [0.6, 0.7], bins = [20,20]
        super().__init__(env)
        self.buckets = [np.linspace(l, h, b-1) for l, h, b in zip(low, high, bins)]
        # [20,20] -> 400
        self.observation_space = gym.spaces.MultiDiscrete(nvec = bins.tolist())
    
    def observation(self, state):
        # [-1.2, 0.] -> (4,3) = (position, velocity)
        indices = tuple(np.digitize(cont, buck) for cont, buck in zip(state, self.buckets))
        return indices
    
bins = np.array([20,20])
low = env.observation_space.low
high = env.observation_space.high
saenv = StateAggregationEnv(env, bins=bins, low=low, high=high)


action_values = np.zeros((20,20,3))

def policy(state, epsilon=0.):
    if np.random.random() < epsilon:
        return np.random.randint(3)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))
    
def sarsa(action_values, 
          policy,
          episodes,
          alpha=0.1,
          gamma=0.99,
          epsilon=0.2):
    stats = {'Returns': []}
    for episode in tqdm(range(1, episodes+1)):
        state = saenv.reset()
        action = policy(state, epsilon)
        done = False
        ep_return = 0

        while not done:
            next_state, reward, done, _ = saenv.step(action)
            next_action = policy(next_state, epsilon)
            qsa = action_values[state][action]
            next_qsa = action_values[next_state][next_action]
            action_values[state][action] = qsa + alpha * (reward + gamma*next_qsa - qsa)
            state = next_state
            action = next_action
            ep_return += reward
        stats['Returns'].append(ep_return)
    return stats

stats = sarsa(action_values, policy, 20000, alpha=0.1, epsilon=0)

plot_stats(stats)
plot_policy(action_values, env.render(mode='rgb_array'), action_meanings={0: 'B', 1: 'N', 2: 'F'})
plot_tabular_cost_to_go(action_values, xlabel='Car Position', ylabel='Velocity')
test_agent(saenv, policy,2)

