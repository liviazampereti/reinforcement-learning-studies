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

env = gym.make('MountainCar-v0')
seed_everything(env)

class TileCodingEnv(gym.ObservationWrapper):

    # Initialize the class

    def __init__(self, env, bins, low, high, n):
        super().__init__(env)
        self.tilings = self._create_tilings(bins, high, low, n)
        self.observation_space = gym.spaces.MultiDiscrete(nvec=bins.tolist() * n)

    # .observation
    def observation(self, state):
        # (5, 4) - (index of the position of the car, index of the velocity of the car)
        # [(5,4), (5,3), (4,4), (5,4)] - list of tuples
        indices = []
        for t in self.tilings:
            tiling_indices = tuple(np.digitize(i, b) for i, b in zip(state, t))
            indices.append(tiling_indices)
        return indices

    # Create tilings
    def _create_tilings(self, bins, high, low, n):
        # [1, 3]
        displacement_vector = np.arange(1, 2*len(bins), 2)
        tillings = []
        for i in range(1, n+1):
            low_i = low - random.random() * 0.2 * low
            high_i = high + random.random() * 0.2 * high
            segment_sizes = (high_i - low_i) / bins
            displacements = displacement_vector * i % n
            displacements = displacements * (segment_sizes / n)
            low_i += displacements
            high_i += displacements
            buckets_i = [np.linspace(l, h, b-1) for l, h, b in zip(low_i, high_i, bins)]
            tillings.append(buckets_i)
        return tillings

tilings = 4
bins = np.array([20,20])
low = env.observation_space.low
high = env.observation_space.high
tcenv = TileCodingEnv(env, bins=bins, low=low, high=high, n=tilings)

action_values = np.zeros((4, 20, 20, 3))

def policy(state, epsilon = 0):
    if np.random.random() < epsilon:
        return np.random.randint(3)
    else:
        av_list = []
        for i, idx in enumerate(state):
            av = action_values[i][idx]
            av_list.append(av)
        # [[1,2,3], [4,5,6]] -> [2.5, 3.5, 4.5]
        av = np.mean(av_list, axis=0)
        return np.random.choice(np.flatnonzero(av==av.max()))
    
def sarsa(action_values, 
          policy,
          episodes,
          alpha=0.1,
          gamma=0.99,
          epsilon=0.2):
    stats = {'Returns': []}
    for episode in tqdm(range(1, episodes+1)):
        state = tcenv.reset()
        action = policy(state, epsilon)
        done = False
        ep_return = 0

        while not done:
            next_state, reward, done, _ = tcenv.step(action)
            next_action = policy(next_state, epsilon)
            for i, (idx, next_idx) in enumerate(zip(state, next_state)):
                qsa = action_values[i][idx][action]
                next_qsa = action_values[i][next_idx][next_action]
                action_values[i][idx][action] = qsa + alpha * (reward + gamma*next_qsa - qsa)
            state = next_state
            action = next_action
            ep_return += reward
        stats['Returns'].append(ep_return)
    return stats

stats = sarsa(action_values, policy, 20000, alpha=0.1, epsilon = 0)

plot_stats(stats)
plot_policy(action_values.mean(axis=0), env.render(mode='rgb_array'), action_meanings = {0: 'B', 1: 'N', 2: 'F'})
plot_tabular_cost_to_go(action_values.mean(axis=0), xlabel='Car Position', ylabel = 'Velocity')
test_agent(tcenv, policy, 2)