import random
import gym
import numpy as np
from discrete_q_learning_agent import DiscreteQLearningAgent
from utils import create_grid, plot_rolling, plot_q_map, plot_route


env = gym.make('MountainCar-v0')
seed = 0
state = env.seed(seed)
random.seed(seed)
np.random.seed(seed)

s_space = env.observation_space
bins = (10, 10)
labels = ['Position', 'Velocity']
grid = create_grid(s_space.low, s_space.high, bins)

# Q-Learning Hyperparamters 
train_episodes = 10_000

alpha = .02
# Testing Parameters
test_episodes = 500
live_episodes = 5

# Discrete q learning agent
agent = DiscreteQLearningAgent(env, grid, q_alpha=alpha)
agent.load_q('MountainCar/q_02')
test_rewards, _ = agent.run(env, episodes = test_episodes, mode ='test')
# Testing Results
avg_reward = np.mean(test_rewards)
print(f"Exploitation Average Reward: {avg_reward}")
plot_rolling(test_rewards, title = 'Test Total Rewards')


# Running Live
for e in range(1, live_episodes + 1):
    state = env.reset()
    score = 0
    state_history = []
    while True:
        action = agent.act(state, mode = 'test')
        state_history.append(agent.last_state)
        env.render()
        state, reward, done, info = env.step(action)
        score += reward
        if done:
            plot_route(bins, state_history, labels,title = f'Live Episode: {e}')
            break
    print('Final score:', score)
env.close()



