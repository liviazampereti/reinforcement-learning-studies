from Maze import Maze, display_video
from matplotlib import pyplot as plt
import numpy as np

env = Maze()
state = env.reset()
done = False
gamma = 0.99
G_0 = 0
t = 0

trajectory = []
frames = []
frames.append(env.render(mode="rgb_array"))

while not done:
    action = env.action_space.sample() #gerados aleatóriamente
    next_state, reward, done, _ = env.step(action)
    G_0 += gamma **t * reward
    t += 1
    trajectory.append([state, action, reward, done, next_state])
    
    img = env.render(mode="rgb_array")
    frames.append(img)

    state = next_state

display_video(frames, 'action_space_sample.mp4')
env.close()

print(f"G0: {G_0} | t: {t}")

#############################################################################
def random_policy(state):
    return np.array([0.25] * 4)


state = env.reset()
done = False
gamma = 0.99
G_0 = 0
t = 0

action_probabilities = random_policy(state)
print(f"Action probabilities: {action_probabilities}")

trajectory = []
frames = []

while not done:
    action = np.random.choice(range(4), 1, p = action_probabilities)
    next_state, reward, done, _ = env.step(action)
    G_0 += gamma **t * reward
    t += 1
    trajectory.append([state, action, reward, done, next_state])
    
    img = env.render(mode="rgb_array")
    frames.append(img)

    state = next_state

display_video(frames, 'random.mp4')
env.close()

print(f"G0: {G_0} | t: {t}")

################################################################################
state = env.reset()
done = False
gamma = 0.99
G_0 = 0
t = 0


trajectory = []
frames = []

while not done:
    action = np.random.choice(range(4), 1, p = [0.125,0.375,0.375,0.125])
    next_state, reward, done, _ = env.step(action)
    G_0 += gamma **t * reward
    t += 1
    trajectory.append([state, action, reward, done, next_state])
    
    img = env.render(mode="rgb_array")
    frames.append(img)

    state = next_state

display_video(frames, 'random_escolha_livia.mp4')
env.close()

print(f"G0: {G_0} | t: {t}")