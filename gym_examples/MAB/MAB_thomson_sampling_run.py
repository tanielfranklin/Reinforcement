import random 
from MAB_env import get_bandit_env_5, thompson_sampling_policy 
import matplotlib.pyplot as plt 
import numpy as np 

def run_thompson_sampling(balance, env, visualize = True): 
    state = env.reset() 
    rewards = [] 
    for i in range(balance): 
        if i ==50: 
            action = thompson_sampling_policy(state, visualize, plot_title = f'Iteration: {i}') 
        elif i % 100 == 0:
            action = thompson_sampling_policy(state, visualize, plot_title = f'Iteration: {i}') 
        else: 
            action = thompson_sampling_policy(state, False, plot_title =f'Iteration: {i}') 
        state, reward, done, debug = env.step(action) 
        rewards.append(reward) 
    env.close() 
    return env, rewards

if __name__ == '__main__': 
    seed = 3 
    random.seed(seed) 
    balance = 1_000 
    env = get_bandit_env_5() 
    env, rewards = run_thompson_sampling(balance, env, visualize =True) 
    env.render() 
    cum_rewards = np.cumsum(rewards) 
    plt.plot(cum_rewards) 
    plt.title('Thompson Sampling Policy') 
    plt.xlabel('Trials') 
    plt.ylabel('Reward') 
    plt.show() 
 
