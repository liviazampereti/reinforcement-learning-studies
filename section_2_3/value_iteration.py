import numpy as np
import matplotlib.pyplot as plt

from Maze1 import Maze
from Maze1 import plot_policy, plot_values, test_agent

import seaborn as sns

env = Maze()
state = env.reset()
env.state = (1, 2)

frame = env.render(mode='rgb_array')

plt.axis('off')
plt.imshow(frame)
plt.show()

# 25 states with 4 possible actions [0.25, 0.25, 0.25, 0.25]
policy_probs = np.full((5, 5, 4), 0.25)

def policy(state):
    return policy_probs[state]

# action_probabilities = policy((0,0))
# for action, prob in zip(range(4), action_probabilities):
#     print(f'Probabilitie of taking action {action}: {prob}')


# test_agent(env, policy, episodes=1)

plot_policy(policy_probs, frame)

state_values = np.zeros(shape=(5,5))

# plot_values(state_values, frame)

#Value Iteration algorithm
def value_iteration(policy_probs, state_values, theta=1e-6, gamma=0.99):
    delta = float('inf')
    episode = 0
    while delta > theta:
        delta = 0
        
        print(f'Episode: {episode}')
        for row in range(5):
            for col in range(5):
                
                
                old_value = state_values[(row, col)]
                action_probs = None
                max_qsa = float('-inf')

                for action in range(4):
                    next_state, reward, _, _ = env.simulate_step((row, col), action)
                    qsa = reward + gamma * state_values[next_state]
                    
                    if qsa > max_qsa:
                        max_qsa = qsa
                        action_probs = np.zeros(4)
                        action_probs[action] = 1.

                state_values[(row,col)] = max_qsa
                policy_probs[(row, col)] = action_probs
                delta = max(delta, abs(max_qsa - old_value))
                plot_values(state_values, frame)
                
        episode += 1

value_iteration(policy_probs, state_values)    
plot_values(state_values, frame)

plot_policy(policy_probs, frame)

test_agent(env, policy)