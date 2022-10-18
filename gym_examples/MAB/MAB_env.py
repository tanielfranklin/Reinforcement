import random 
import gym 
import numpy as np 
from scipy.stats import beta 
import matplotlib.pyplot as plt 
class MultiArmedBanditEnv(gym.Env): 
    def __init__(self, bandits): 
        self.bandits = bandits # number of machines
        self.state = {} # cumulative earnings of each machine
        self.reset() 

    def step(self, action): 
        p = self.bandits[action] 
        r = random.random() 
        reward = 1 if r <= p else -1 
        self.state[action].append(reward) 
        done = False 
        debug = None 
        return self.state, reward, done, debug 
    def reset(self): 
        self.state = {}
        for i in range(len(self.bandits)): 
            self.state[i] = [] 
        return self.state 
    def render(self, mode = "ascii"): 
        returns = {} 
        trials = {} 
        for i in range(len(self.bandits)): 
            returns[i] = sum(self.state[i]) 
            trials[i] = len(self.state[i]) 
        print(f'=====Total Trials: {sum(trials.values())}=====') 
        for b, r in returns.items(): 
            t = trials[b] 
            print(f'Bandit {b}| returns: {r}, trials: {t}') 
        print(f'=====Total Returns: {sum(returns.values())}=====')
def get_bandit_env_5(): 
    bandits = [.45, .45, .4, .6, .4] # Five machines with probabilities
    return MultiArmedBanditEnv(bandits)

def greedy_policy(state, explore=10):
    """
    It first loses money while completing the  exploration process,
    but then the returns rise in the exploitation mode
    """
    bandits = len(state)
    trials = sum([len(state[b]) for b in range(bandits)])
    total_explore_trials = bandits * explore
    # exploration (investigating machine with better earning probabilities )

    if trials <= total_explore_trials:
        return trials % bandits #Each machine has a time to be used
    # exploitation(Choosing that machine with highest earning probabilities)
    avg_rewards = [sum(state[b]) / len(state[b]) for b in
                range(bandits)]
    best_bandit = np.argmax(avg_rewards)
    return best_bandit

def e_greedy_policy(state, explore = 10, epsilon = .1):
    """
    After first exploration it explores with probability epsilon and acts as greedy
    policy with probability 1-epsilon (exploitation)
    """ 
    bandits = len(state) 
    trials = sum([len(state[b]) for b in range(bandits)]) 
    total_explore_trials = bandits * explore 
    # exploration 
    if trials <= total_explore_trials: 
        return trials % bandits
    # random bandit 
    if random.random() < epsilon: 
        return random.randint(0, bandits - 1) # return an int of 0 to bandits-1
    # exploitation (greedy policy)
    avg_rewards = [sum(state[b]) / len(state[b]) for b in range(bandits)] 
    best_bandit = np.argmax(avg_rewards) 
    return best_bandit 



def thompson_sampling_policy(state, visualize = False, plot_title= ''): 
    action = None 
    max_bound = 0 
    color_list = ['red', 'blue', 'green', 'black', 'yellow']
    for b, trials in state.items(): 
        w = len([r for r in trials if r == 1]) 
        l = len([r for r in trials if r == -1]) 
        if w + l == 0: 
            avg = 0 #avoid division by 0
        else: 
            avg = round(w / (w + l), 2) # Distribution average
        random_beta = np.random.beta(w + 1, l + 1) # Value obtained with this distributions
        if random_beta > max_bound: 
            max_bound = random_beta # The highest value will define the bandit to be used
            action = b
        if visualize: 
            color = color_list[b % len(color_list)]
            #Domain of each distribution 
            x = np.linspace(beta.ppf(0.01, w, l), beta.ppf(0.99, w, l), 100) 
            plt.plot(x, beta.pdf(x, w, l), 
            label = f'Bandit {b}| avg={avg}, v={round(random_beta,2)}', 
            color = color, linewidth = 3) 
            plt.axvline(x = random_beta, color = color, linestyle = '--') 
    if visualize: 
        plt.title('Thompson Sampling: Beta Distribution. ' + plot_title) 
        plt.legend() 
        plt.show() 
    return action

def run_e_greedy_policy(balance, env, exploration = 10, epsilon =.1): 
    state = env.reset() 
    rewards = [] 
    for i in range(balance): 
        action = e_greedy_policy(state, exploration, epsilon) 
        state, reward, done, debug = env.step(action) 
        rewards.append(reward) 
    env.close() 
    return env, rewards 

def run_thompson_sampling(balance, env, visualize=True):
    state = env.reset()
    rewards = []
    for i in range(balance):
        if i == 50:
            action = thompson_sampling_policy(
                state, visualize, plot_title=f'Iteration: {i}')
        elif i % 100 == 0:
            action = thompson_sampling_policy(
                state, visualize, plot_title=f'Iteration: {i}')
        else:
            action = thompson_sampling_policy(
                state, False, plot_title=f'Iteration: {i}')
        state, reward, done, debug = env.step(action)
        rewards.append(reward)
    env.close()
    return env, rewards 
