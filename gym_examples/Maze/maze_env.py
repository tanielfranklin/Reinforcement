import gym
import matplotlib.pyplot as plt
import numpy as np
import random

import matplotlib.pyplot as plt
import numpy as np

def plot_q_map(x_coord, y_coord, blocks, Q, goal, title = None):
    nrows = len(y_coord)
    ncols = len(x_coord)
    image = np.zeros(nrows * ncols)
    image = image.reshape((nrows, ncols))

    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            if label in blocks:
                image[y, x] = 0
            elif label == goal:
                image[y, x] = .5
            else:
                image[y, x] = 1

    plt.figure(figsize = (nrows, ncols), dpi = 240)
    plt.matshow(image, cmap = 'gray', fignum = 1)

    states = maze_states(x_coord, y_coord)
    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            state_idx = states.index(label)
            for a_idx in range(len(actions)):
                is_max = np.argmax(Q[state_idx]) == a_idx
                if is_max:
                    arrow_color = 'green'
                    font_size = 10
                    arrow_len = .04
                else:
                    arrow_color = 'pink'
                    font_size = 7
                    arrow_len = .03

                q_v = round(Q[state_idx, a_idx])
                if q_v == 0:
                    continue

                if actions[a_idx] == 'D':
                    arrow_x, arrow_y = 0, arrow_len
                    annotate_xy = (x, y + .3)
                elif actions[a_idx] == 'U':
                    arrow_x, arrow_y = 0, -arrow_len
                    annotate_xy = (x, y - .2)
                elif actions[a_idx] == 'R':
                    arrow_x, arrow_y = arrow_len, 0
                    annotate_xy = (x + .25, y)
                elif actions[a_idx] == 'L':
                    arrow_x, arrow_y = -arrow_len, 0
                    annotate_xy = (x - .4, y - .1)

                plt.annotate(q_v, xy = annotate_xy, fontsize = font_size)
                plt.arrow(x, y, arrow_x, arrow_y, width = arrow_len, color = arrow_color)

    plt.xticks(range(ncols), x_coord)
    plt.yticks(range(nrows), y_coord)
    if title:
        plt.title(title, pad = 12)
    plt.show()

def map1():
    x_coord = ['A', 'B', 'C', 'D']
    y_coord = [1, 2, 3, 4, 5]
    walls = ['B3', 'B4', 'C2', 'C4', 'D2']
    goal = 'C3'
    return x_coord, y_coord, walls, goal


# Up, Right, Down, Left - Actions in tthe map
actions = ['U', 'R', 'D', 'L']


def plot_maze_state(x_coord, y_coord, walls, current_state, goal, title=''):
    nrows = len(y_coord)
    ncols = len(x_coord)
    image = np.zeros((nrows) * ncols)
    image = image.reshape((nrows, ncols))
    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            if label in walls:
                image[y, x] = 0
            else:
                image[y, x] = 1
    plt.figure(figsize=(ncols, nrows+4), dpi=120)
    plt.matshow(image, cmap='gray', fignum=1)
    for x in range(ncols):
        for y in range(nrows):
            label = f'{x_coord[x]}{y_coord[y]}'
            if label == goal:
                plt.annotate('O', xy=(x - .2, y + .2), fontsize=20, weight='bold')
            if label == current_state:
                plt.annotate('X', xy=(x - .2, y + .2), fontsize=20, weight='bold')
    plt.xticks(range(ncols), x_coord)
    plt.yticks(range(nrows), y_coord)
    if title:
        plt.title(title)
    plt.show()
            
            
def move(current_state, action, x_coord, y_coord, walls): 
    x = current_state[0] 
    y = int(current_state[1:]) 
    x_idx = x_coord.index(x) 
    y_idx = y_coord.index(y) 
    if action == 'U': 
        next_state = x + str(y_coord[max([y_idx - 1, 0])]) 
    elif action == 'R': 
        next_state = x_coord[min([x_idx + 1, len(x_coord) - 1])] + str(y) 
    elif action == 'D': 
        next_state = x + str(y_coord[min([y_idx + 1, len(y_coord) - 1])]) 
    elif action == 'L': 
        next_state = x_coord[max([x_idx - 1, 0])] + str(y) 
    else: 
        raise Exception(f'Invalid action: {action}') 
    if next_state in walls: 
        return current_state 
    return next_state

def random_state(x_coord, y_coord, walls,seed):
    available_states = []
    random.seed(seed)
    for cell in maze_states(x_coord, y_coord):
        if cell in walls:
            continue
        available_states.append(cell)
    return random.choice(available_states)

def maze_states(x_coord, y_coord):
    all_states = []
    for y in y_coord:
        for x in x_coord:
            cell = f'{x}{y}'
            all_states.append(cell)
    return all_states


def epsilon_greedy_policy(e, Q, s):
    #greedy with probability e
    # and 
    r = random.random()
    if r > e:
        max_q = max(Q[s]) # Maximun q for the given state
        # Select actions if actions == max value
        candidates = [i for i in range(len(Q[s, :])) if Q[s, i] == max_q]
        # Select randomly among the candidates
        return random.choice(candidates)
    else:
        return random.randint(0, len(Q[s]) - 1)
    
def q_learning(Q, current_s, next_s, a, r, gamma = .9, alpha =.5):
    # r + g*max[a](Q(S', a)) - Q(S, A)
    td = (r + gamma * max(Q[next_s]) - Q[current_s, a]) #temporal difference
    Q[current_s, a] += alpha * round(td, 2)
    return Q




class MazeEnv(gym.Env):
    def __init__(self, x_coord, y_coord, blocks, finish_state):
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.blocks = blocks
        self.state = random_state(x_coord, y_coord, blocks,None)
        self.finish_state = finish_state
        # debug properties
        self._total_actions = 0
        self._total_reward = 0
    def step(self, action):
        prev_state = self.state
        next_state = move(self.state, action, self.x_coord,self.y_coord, self.blocks)
        reward = 0
        done = False
        if next_state == self.finish_state:
            reward = 1000
            done = True
        elif prev_state == next_state:
            reward = -10
        elif prev_state != next_state:
            reward = -1
        self.state = next_state
        self._total_actions += 1
        self._total_reward += reward
        return self.state, reward, done, None
    def reset(self,seed=None):
        
        self._total_actions = 0
        self._total_reward = 0
        not_allowed = self.blocks + [self.finish_state]
        self.state = random_state(self.x_coord, self.y_coord,not_allowed,seed)
        return self.state
    def render(self, mode = "ascii"):
        plot_maze_state(self.x_coord,self.y_coord,
        self.blocks, self.state, self.finish_state,
        f'Total Reward: {self._total_reward} \n'
        f'Total Actions: {self._total_actions}')
    def maze_states(x_coord, y_coord):
        all_states = []
        for y in y_coord:
            for x in x_coord:
                cell = f'{x}{y}'
                all_states.append(cell)
        return all_states




