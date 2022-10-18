import numpy as np

from collections import defaultdict
def monte_carlo_policy(env): 
    state_history = [] 
    state = env.reset() 
    #stick 0 and hit:1
    actions = [0, 1] 
    while True: 
        player_sum = state[0] 
        if player_sum < 12: 
            action = 1 #Hit
        else: 
            #80% stick and 20% hit otherwise invert
            probs = [0.8, 0.2] if player_sum > 17 else [0.2, 0.8]
            #choose a random action with defined probabilities 
            action = np.random.choice(actions, p = probs)
        #observe env return
        next_state, reward, terminated, truncated,_= env.step(action)
        #stores results 
        state_history.append([state, action, reward, next_state])
        #update state 
        state = next_state 
        if (terminated or truncated): 
            break 
    return state_history



def q_greedy_policy(env, Q):
    state = env.reset()
    while True:
        agent_sum = state[0]
        if agent_sum < 12:
            action = 1
        else:
            action = np.argmax(Q[state])
        next_state, reward, terminated, truncated,_ = env.step(action)

        if (terminated or truncated):
            break

        state = next_state

    return reward


def train_q_monte_carlo(env, train_episodes, gamma=1):
    G = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    for e in range(1, train_episodes + 1):
        episode = monte_carlo_policy(env)
        # unpack MCL history
        states, actions, rewards, next_states = zip(*episode)
        # build discount array
        discounts = np.array([gamma**i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            # sum weighted (or discounted) future rewards
            g = sum(rewards[i:] * discounts[:-(1 + i)])
            G[state][actions[i]] += g
            N[state][actions[i]] += 1.0  # Count actions
            # update position in Q
            Q[state][actions[i]] = G[state][actions[i]] / N[state][actions[i]]
        if e % 10_000 == 0:  # stop if it reaches 10000
            print(f'Episodes: {e}/{train_episodes}')
    return Q 
