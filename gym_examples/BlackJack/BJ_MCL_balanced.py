import gym
import random

from MCL_policy import q_greedy_policy, train_q_monte_carlo
from plot_q import plot_q

#Monte Carlo Exploration and Greed Exploitation

if __name__ == '__main__':
    env = gym.make('Blackjack-v1', new_step_api=True)
    seed = 0
    random.seed(seed)
    env.reset(seed=seed)
    Q = train_q_monte_carlo(env, 50_000)
    plot_q(Q) 

    # print some values
    episodes = 1000 
    wins = 0 
    loss = 0 
    draws = 0 
    for e in range(1, episodes + 1): 
        reward = q_greedy_policy(env, Q) 
        if reward > 0: 
            wins += 1 
        elif reward < 0: 
            loss += 1 
        else: 
            draws += 1 
    print(f'Wins: {wins} | Loss: {loss} | Draws: {draws}')

