import os
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW


from utils import test_policy_network, seed_everything, plot_stats, plot_action_probs


class PreprocessEnv(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        state = self.env.reset()
        return torch.from_numpy(state).float()

    def step_async(self, actions):
        actions = actions.squeeze().numpy()
        self.venv.step_async(actions)

    def step(self, actions):
        actions = actions.squeeze().numpy()
        next_state, reward, done, info = self.env.step(actions)
        next_state = torch.from_numpy(next_state).float()
        reward = torch.tensor(reward).unsqueeze(1).float()
        done = torch.tensor(done).unsqueeze(1)
        return next_state, reward, done, info
    
env = gym.make('CartPole-v0')
dims = env.observation_space.shape[0]
actions = env.action_space.n

print(f'State dims: {dims} | Actions: {actions}')
print(f'Sample state: {env.reset()}')

num_envs = os.cpu_count()
print(f'Number of envs: {num_envs}')
parallel_env = gym.vector.make('CartPole-v0', num_envs=num_envs)
seed_everything(parallel_env)
parallel_env = PreprocessEnv(parallel_env)   

policy = nn.Sequential(
    nn.Linear(dims, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64,actions),
    nn.Softmax(dim=-1)
)

neutral_state = torch.zeros(4)
left_danger = torch.tensor([-2.3, 0, 0, 0])
right_danger = torch.tensor([2.3, 0, 0, 0])


def reinforce(policy, episodes, alpha=1e-4, gamma=0.99):
    optim = AdamW(policy.parameters(), lr = alpha)
    stats = {'Loss': [], 'Returns': []}

    for episode in tqdm(range(1, episodes + 1)):
        state = parallel_env.reset()
        done_b = torch.zeros((num_envs,1), dtype=torch.bool)
        transitions = []
        ep_return = torch.zeros((num_envs, 1))
        
        while not done_b.all():
            action = policy(state).multinomial(1).detach()
            next_state, reward, done, _ = parallel_env.step(action)
            transitions.append([state, action, ~done_b * reward])
            ep_return += reward
            done_b |= done
            state = next_state

        G = torch.zeros((num_envs,1))
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + gamma*G
            probs_t = policy(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)
            
            entropy_t = -torch.sum(probs_t * log_probs_t, dim=-1, keepdim = True)
            gamma_t = gamma **t

            pg_loss_t = - gamma_t * G * action_log_prob_t - entropy_t
            total_loss_t = (pg_loss_t - 0.01 * entropy_t).mean()

            policy.zero_grad()
            total_loss_t.backward()
            optim.step()
        stats['Loss'].append(total_loss_t.item())
        stats['Returns'].append(ep_return.mean().item())

    return stats

parallel_env.reset()
stats = reinforce(policy, 200)
plot_stats(stats)
test_policy_network(env, policy, episodes=5)