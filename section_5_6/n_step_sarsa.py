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

def n_step_sarsa(action_values, policy, 
                 episodes,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.2,
                 n=8):
    for episode in tqdm(range(1, episodes+1)):
        state = env.reset()
        action = policy(state, epsilon)
        transitions = []
        done = False
        t = 0
        
        while t-n<len(transitions):
            # Execute an action in the environment

            if not done:
                next_state, reward, done, _ = env.step(action)
                next_action = policy(next_state, epsilon)
                transitions.append([state, action, reward])

            # Update de q-value estimates
            if t>=n:
                # G = r1 + gamma*r2 + gamma^2*r3 + .... +  gamma^n * Q(Sn, An)
                G = (1 - done) * action_values[next_state][next_action]
                for state_t, action_t, reward_t in reversed(transitions[t-n:]):
                    G = reward_t + gamma * G
                
                action_values[state_t][action_t] += alpha * (G - action_values[state_t][action_t])

            t+=1
            state = next_state
            action = next_action


n_step_sarsa(action_values, policy, episodes = 1000)
plot_action_values(action_values)
plot_policy(action_values, env.render(mode='rgb_array'))
test_agent(env, policy)

            