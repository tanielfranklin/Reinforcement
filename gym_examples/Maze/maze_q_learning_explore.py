import random
import numpy as np
from maze_env import MazeEnv, maze_states, map1, epsilon_greedy_policy, q_learning, plot_q_map

actions = ['U', 'R', 'D', 'L']
x_coord, y_coord, blocks, goal = map1()
seed = 1
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)
env.reset(seed=seed)
states = maze_states(x_coord, y_coord)

# Global parameters
epsilon = .2
gamma = 0.9
alpha = 0.8
episodes = 50

# Initializing Q
Q = np.array(np.zeros([len(states), len(actions)]))


for e in range(episodes):
    state = env.reset()
    i = 0

    while True:
        i+=1
        # Next state by e-greedy policy
        action_idx = epsilon_greedy_policy(epsilon, Q, states.index(state))
        action = actions[action_idx]
        # perform the action selected
        next_state, reward, done, debug = env.step(action)
        # Update Q
        Q = q_learning(
            Q,
            states.index(state),
            states.index(next_state),
            action_idx,
            reward,
            gamma,
            alpha
        )
        state = next_state
        if done:
            if e < 5 or e % 10 == 9 or e == episodes - 1:
                plot_q_map(x_coord, y_coord, blocks, Q,
                        goal, title=f'After Final Action')
            break
    env.close()
    """Q-learning back-propagates rewards across previous states. Thus, a particular
route of action is created, which will lead to the greatest total reward in various
situations."""
        
 
