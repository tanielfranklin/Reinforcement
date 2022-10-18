import random
import numpy as np
from maze_env import MazeEnv,maze_states,map1, epsilon_greedy_policy,q_learning, plot_q_map

actions = ['U', 'R', 'D', 'L']
x_coord, y_coord, blocks, goal = map1()
seed = 10
random.seed(seed)
env = MazeEnv(x_coord, y_coord, blocks, goal)
env.reset(seed=seed)
states = maze_states(x_coord, y_coord)

#Global parameters
epsilon = .2
gamma = 0.9
alpha = 0.8
# Initializing Q
Q = np.array(np.zeros([len(states), len(actions)]))
state = env.reset()

i = 1
while True:
    # Next state by e-greedy policy
    action_idx = epsilon_greedy_policy(epsilon, Q,states.index(state))
    action = actions[action_idx]
    #perform the action selected
    next_state, reward, done, debug = env.step(action)
    #Update Q
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
        plot_q_map(x_coord, y_coord, blocks, Q, goal,title = f'After Final Action')
        break
    else:
        if i % 10 == 0 or i < 3:
            plot_q_map(x_coord, y_coord, blocks, Q, goal,title = f'After Action: {i}')
    i += 1






