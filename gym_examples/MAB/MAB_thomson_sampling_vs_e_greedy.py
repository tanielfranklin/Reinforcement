import random
from MAB_env import get_bandit_env_5, run_thompson_sampling, run_e_greedy_policy
import matplotlib.pyplot as plt
import numpy as np





if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    episodes = 50 
    balance = 1_000 
    env_gen = get_bandit_env_5 
    rewards = { 
    'Epsilon Greedy': [np.cumsum(run_e_greedy_policy(balance,
    env_gen())[1]) for _ in range(episodes)], 
    'Thompson Sampling': [np.cumsum(run_thompson_sampling(balance,
    env_gen(),visualize=False)[1]) for _ in range(episodes)], 
    } 
    for policy, r in rewards.items(): 
        plt.plot(np.average(r, axis = 0), label = policy) 
        plt.legend() 
        plt.xlabel("Rounds") 
        plt.ylabel("Average Returns") 
        plt.title('Battle') 
    plt.show() 
