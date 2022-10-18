import gym
from utils import discretize, create_grid, plot_route
import numpy as np
import random

env = gym.make('MountainCar-v0', render_mode='human', new_step_api=True)
seed = 0
state=env.reset(seed=seed)
random.seed(seed)
np.random.seed(seed)
s_space = env.observation_space

#Create discretization space
bins = (10, 10)
grid = create_grid(s_space.low, s_space.high, bins)

state_history = []
d_state = discretize(env.reset(), grid)
state_history.append(d_state)
while True:
    env.render()
    random_action = env.action_space.sample()
    state, reward, terminated, truncated,_ = env.step(random_action)
    d_state = discretize(state, grid)
    state_history.append(d_state)
    print(d_state)
    if (terminated or truncated):
        break
env.close()
