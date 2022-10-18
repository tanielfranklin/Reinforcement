import os 
import random 
import gym 
import numpy as np 
import matplotlib.pyplot as plt 
from dqn_agent import DqnPtAgent 
cwd = os.path.dirname(os.path.abspath(__file__))
env = gym.make('LunarLander-v2') 
seed = 1 
random.seed(seed) 
env.reset() 

print(env.reset() )
exit()

from replay_buffer import ReplayBuffer
# rp=ReplayBuffer(100,10)
# print(rp.memory.batch())
# exit()

# PyTorch Implementation 
agent = DqnPtAgent(state_size = 8, action_size = 4) 
save_path = cwd + '/dqn_pt_agent.pth' 
episodes = 500 
scores = [] 
for e in range(1, episodes + 1): 
    state = env.reset() 
    score = 0
    agent.before_episode()
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done) 
        state = next_state 
        score += reward
        if done: 
            break
        
    scores.append(score) 
    if e % 10 == 0: 
        print(f'Episode {e} Average Score: {np.mean(scores[-100:])}')
agent.save(save_path) 
#Training results: 
plt.plot(np.arange(len(scores)), scores) 
plt.ylabel('Score') 
plt.xlabel('Episode') 
plt.show()  