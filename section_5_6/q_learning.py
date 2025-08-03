import numpy as np
import matplotlib.pyplot as plt

from Maze import Maze
from Maze import plot_policy, plot_action_values, test_agent

from tqdm import tqdm 

env = Maze()

action_values = np.zeros((5,5,4))

def target_policy(state):
    av = action_values[state]
    return np.random.choice(np.flatnonzero(av==av.max()))

def exploratory_policy(state):
    return np.random.randint(4)

def q_learning(action_values,
               exploratory_policy,
               target_policy,
               episodes,
               alpha=0.1,
               gamma=0.99):
    for episode in tqdm(range(1, episodes+1)):
        state = env.reset()
        done = False

        while not done:
            action = exploratory_policy(state)
            
            next_state, reward, done, _ = env.step(action)
            next_action = target_policy(next_state)
            
            qsa = action_values[state][action]
            next_qsa = action_values[next_state][next_action]

            action_values[state][action]=qsa + alpha*(reward + gamma*next_qsa - qsa)

            state = next_state

q_learning(action_values, exploratory_policy, target_policy, 1000)
plot_action_values(action_values)
plot_policy(action_values, env.render(mode='rgb_array'))
test_agent(env, target_policy)

            