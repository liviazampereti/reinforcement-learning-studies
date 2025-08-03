import random
import copy
import gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from Maze import plot_cost_to_go, plot_max_q, test_agent, plot_stats, seed_everything

# Create the environment

env = gym.make('MountainCar-v0')
seed_everything(env)
state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n 
print(f'Mountain Car env: \nState dimensions: {state_dims} (position, velocity)\nNumber of actions: {num_actions} (left, nothing, right)\n')

# Prepare the environment to work with PyTorch
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    # Wraps env.reset
    def reset(self):
        state = self.env.reset()
        # [[0,0], [1,1], [N X D]], where N = number of observations, D = number of dimensions
        return torch.from_numpy(state).unsqueeze(dim = 0).float()

    # Wraps env.step
    def step(self, action):
        action = action.item()
        next_state, reward, done, info = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze(dim = 0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done, info
    
env = PreprocessEnv(env)
state = env.reset()
action = torch.tensor(0)
next_state, reward, done, _ = env.step(action)
print(f'Sample state: {state}')
print(f'Next state: {next_state} | Reward: {reward} | Done: {done}')

# Create the Q-Network 
q_network = nn.Sequential(
    nn.Linear(state_dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
)

# Create the target Q-Network
target_q_network = copy.deepcopy(q_network)
target_q_network = target_q_network.eval()

print(q_network)
print(target_q_network)

# Create the e-greedy policy for PyTorch
def policy(state, epsilon=0.):    
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        av = q_network(state).detach()
        return torch.argmax(av, dim=-1, keepdim=True)
    
# Plot the cost to go
#plot_cost_to_go(env, q_network, xlabel = 'Car Position', ylabel='Car Velocity')

# Create the Experience Replay Buffer
class ReplayMemory:
    def __init__(self, capacity = 1000000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # insert: [s, a, r, s']
    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position +1) % self.capacity

    # sample [[s, a, r, s'], [s, a, r, s']]
    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        #[[s, a, r, s'], [s, a, r, s'], [s, a, r, s']] -> [[s, s, s], [a, a, a], [r, r, r], [s', s', s']]
        batch = zip(*batch)
        return [torch.cat(items) for items in batch] # N X D

    # can_sample -> True/False
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size *10

    # __len__
    def __len__(self):
        return len(self.memory)
    

def deep_sarsa(q_network, 
               policy, 
               episodes, 
               alpha = 0.001, 
               batch_size = 32, 
               gamma = 0.99, 
               epsilon = 0.05):
    optim = AdamW(q_network.parameters(), lr = alpha)
    memory = ReplayMemory(capacity=1000000)
    stats = {'MSE Loss': [],
             'Returns': []}
    for episode in tqdm(range(1, episodes +1)):
        state = env.reset()
        done = False
        ep_return = 0

        while not done:
            action = policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            memory.insert([state, action, reward, done, next_state])

            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                qsa_b = q_network(state_b).gather(1, action_b)
                next_action_b = policy(next_state_b, epsilon)
                next_qsa_b = target_q_network(next_state_b).gather(1, next_action_b)
                target_b = reward_b + ~done_b * gamma * next_qsa_b
                
                loss = F.mse_loss(qsa_b, target_b)
                q_network.zero_grad()
                loss.backward()
                optim.step()
                
                stats['MSE Loss'].append(loss.item())

            state = next_state
            ep_return += reward.item()

        stats['Returns'].append(ep_return)
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

    return stats

stats = deep_sarsa(q_network, policy, 2500, epsilon = 0.01)

plot_stats(stats)
plot_cost_to_go(env, q_network, xlabel = 'Car Position', ylabel='Car Velocity')
plot_max_q(env, q_network, xlabel='Car Position', ylabel='Velocity',
           action_labels=['Back', 'Do nothing', 'Forward'])

test_agent(env, policy, episodes=2)