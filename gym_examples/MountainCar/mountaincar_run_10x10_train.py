import random
import gym
import numpy as np
from discrete_q_learning_agent import DiscreteQLearningAgent
from utils import create_grid, plot_rolling, plot_q_map,plot_route

env = gym.make('MountainCar-v0', render_mode='human', new_step_api=True)
env = gym.make('MountainCar-v0', new_step_api=True)
seed = 0
state=env.reset(seed=seed)
random.seed(seed)
np.random.seed(seed)

s_space = env.observation_space
bins = (10, 10)
labels = ['Position', 'Velocity']
grid = create_grid(s_space.low, s_space.high, bins)

# Q-Learning Hyperparamters
train_episodes = 10_000
alpha = .02


#Discrete q learning agent
agent = DiscreteQLearningAgent(env, grid, q_alpha = alpha)
# agent.load_q('MountainCar/q_01')
# train_rewards, d_changes = agent.run(env, episodes=train_episodes)
# agent.save_q('MountainCar/q_02')

train_rewards, d_changes = agent.run(env, episodes=train_episodes)
# Training Results
print("Training finished")
plot_rolling(train_rewards, title = 'Train Total Rewards')
plot_rolling(d_changes, title = 'Train Direction Changes')
# Q(s,a) visualization
plot_q_map(agent.q, labels, title = 'Q Map')



