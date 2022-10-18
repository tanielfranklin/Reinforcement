import matplotlib.pyplot as plt
import random
import numpy as np
from maze_env import actions,map1,plot_maze_state,move



if __name__ == '__main__':
    x_coord, y_coord, walls, goal = map1()
    state = 'A1'
    plot_maze_state(x_coord, y_coord, walls, state, goal,title='')
    actions = ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'U', 'U', 'L']
    for action in actions:
        state = move(state, action, x_coord, y_coord, walls)
    plot_maze_state(x_coord, y_coord, walls, state, goal,title=f'Total Reward:  \n'
        f'Total Actions: ')





 


