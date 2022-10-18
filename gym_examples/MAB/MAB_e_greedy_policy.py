import numpy as np
import random
import matplotlib.pyplot as plt
from MAB_env import get_bandit_env_5, e_greedy_policy


def run_e_greedy_policy(balance, env, exploration = 10, epsilon =.1): 
    state = env.reset() 
    rewards = [] 
    for i in range(balance): 
        action = e_greedy_policy(state, exploration, epsilon) 
        state, reward, done, debug = env.step(action) 
        rewards.append(reward) 
    env.close() 
    return env, rewards 


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    balance = 1_000
    env = get_bandit_env_5()
    env, rewards = run_e_greedy_policy(balance, env)
    env.render()
    cum_rewards = np.cumsum(rewards)
    plt.plot(cum_rewards)
    plt.title('Greedy Policy')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.show()
