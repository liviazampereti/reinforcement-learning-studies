import numpy as np
import matplotlib.pyplot as plt

from Maze import Maze
from Maze import plot_policy, plot_action_values, test_agent

from tqdm import tqdm 

env = Maze()

action_values = np.zeros((5,5,4))

def policy(state, epsilon = 0):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av==av.max()))
    
def sarsa(action_values, 
          policy,
          episodes,
          alpha=0.1,
          gamma=0.99,
          epsilon=0.2):
    
    for episode in tqdm(range(1, episodes+1)):
        state = env.reset()
        action = policy(state, epsilon)
        done = False

        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state, epsilon)
            qsa = action_values[state][action]
            next_qsa = action_values[next_state][next_action]
            action_values[state][action] = qsa + alpha * (reward + gamma*next_qsa - qsa)
            state = next_state
            action = next_action

sarsa(action_values, policy, 150000)
plot_action_values(action_values)
plot_policy(action_values, env.render(mode='rgb_array'))
test_agent(env, policy)