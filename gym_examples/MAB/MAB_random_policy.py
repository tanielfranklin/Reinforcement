import random 
import gym 
import numpy as np 
from MAB_env import get_bandit_env_5


    
if __name__ == '__main__': 
    seed = 1 # freeze random 
    random.seed(seed) 
    np.random.seed(1) 
    balance = 1_000 # same that 1000
    env = get_bandit_env_5() 
    state = env.reset() 
    rewards = [] 
    for i in range(balance): 
        random_bandit = random.randint(0, 4) 
        state, reward, done, debug = env.step(random_bandit) # choose an action randomly
        rewards.append(reward) #log the reward
    env.render() 
    env.close()  

