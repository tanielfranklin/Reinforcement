import torch
import json
import random
import torch.nn as nn
import numpy as np
from tqdm import trange
from IPython.display import clear_output
from IPython.display import HTML
import data_panda as rbt
from torch.utils.tensorboard import SummaryWriter
import glob
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def clear():
    os.system('clear')


state_shape = 3
env = rbt.Panda_RL()
agent = rbt.DQNAgent(state_shape, epsilon=0).to(device)
env.delta = 0.02
RESTORE_AGENT = False  # Restore a trained agent
NEW_BUFFER = True  # Restore a buffer
TRAIN = True  # Train or only simulate
env.renderize = False  # stop robot viewing

agent.n_layer1=128
agent.n_layer1=256


target_network = rbt.DQNAgent(agent.state_shape, epsilon=0.5).to(device)
# Copying weights from agent network
target_network.load_state_dict(agent.state_dict())

# set a seed
seed = 13
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Fill buffer with samples collected ramdomly from environment
buffer_len = 5000
tmax = 800
env.renderize = False

exp_replay = rbt.PrioritizedReplayBuffer(buffer_len)


print("Building buffer")
for i in trange(100,desc="Buuffering"):

    state = env.reset()
    # Play 100 runs of experience with 100 steps and  stop if reach 10**4 samples
    rbt.play_and_record(state, agent, env, exp_replay, n_steps=60)

    if len(exp_replay) == buffer_len:
        break
print(len(exp_replay))

RES_DIR = rbt.set_res_dir()
comment = ""
# monitor_tensorboard()
tb = SummaryWriter(log_dir=RES_DIR, comment=comment)

LOAD_MODEL=False
if LOAD_MODEL:
    agent.load_weights("results_8")
    percentage_of_total_steps=0.4   

# setup some parameters for training
percentage_of_total_steps = 0.80
timesteps_per_epoch = 1
batch_size = 64
total_steps = 50 * 10**3
#total_steps = 10

# init Optimizer
lr = 1e-4
opt = torch.optim.Adam(agent.parameters(), lr=lr)

# set exploration epsilon
start_epsilon = 1
#start_epsilon = 0.1
end_epsilon = 0.05
eps_decay_final_step = percentage_of_total_steps*total_steps

# setup some frequency for logging and updating target network
loss_freq = 40
refresh_target_network_freq = 50
eval_freq = 1000

# to clip the gradients
max_grad_norm = 5000




hyperparameters_train = {"start_epsilon": start_epsilon,
                         "end_epsilon": end_epsilon,
                         "lr": lr,
                         "batch_size": batch_size,
                         "total_steps": total_steps,
                         "percentage_of_total_steps": percentage_of_total_steps,
                         "refresh_target_network_freq": refresh_target_network_freq,
                         "buffer_len": buffer_len,
                         "tmax": tmax,
                         "agent": str(agent.network)
                         }


def save_hyperparameter(dict, directory):
    with open(directory+"/hyperparameters.json", "w") as outfile:
        json.dump(dict, outfile)


# Start training
state = env.reset()
tb.add_graph(agent.network, torch.tensor(
    state, device=device, dtype=torch.float32))
save_hyperparameter(hyperparameters_train, RES_DIR)
loss_min = np.inf
rw_min=-np.inf
print(f"Frequency evaluation = {eval_freq}")
print(f"buffer size = {len(exp_replay)} ")   
print(RES_DIR,device)

 

for step in trange(total_steps + 1, desc="Training"):


    # reduce exploration as we progress
    agent.epsilon = rbt.epsilon_schedule(
        start_epsilon, end_epsilon, step, eps_decay_final_step)

    # take timesteps_per_epoch and update experience replay buffer
    _, state = rbt.play_and_record(
        state, agent, env, exp_replay, timesteps_per_epoch)

    # train by sampling batch_size of data from experience replay
    states, actions, rewards, next_states, done_flags, weights, idxs = exp_replay.sample(
        batch_size)
    actions = [agent.get_action_index(i) for i in actions]

    # loss = <compute TD loss>

    loss = rbt.compute_td_loss_priority_replay(agent, target_network, exp_replay,
                                               states, actions, rewards, next_states, done_flags, weights, idxs,
                                               gamma=0.99,
                                               device=device)
    if loss < loss_min:
        torch.save(agent.state_dict(), RES_DIR+'/best-model-loss.pt')
        loss_min=loss

    tb.add_scalar("1/TD Loss", loss, step)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    opt.step()
    opt.zero_grad()

    tb.add_scalar("2/Epsilon", agent.epsilon, step)

    if step % refresh_target_network_freq == 0:
        # Load agent weights into target_network
        target_network.load_state_dict(agent.state_dict())

    if step % eval_freq == 0:
        # eval the agent

        #clear_output(True)        
        m_reward = rbt.evaluate(env, agent, n_games=50,
                                greedy=True, t_max=tmax)[0]
        tb.add_scalar("1/Mean reward per episode", m_reward, step)
        #print(f"Last mean reward = {m_reward}")
        
        

        
        
    if m_reward > rw_min:
        torch.save(agent.state_dict(), RES_DIR+'/best-model-rw.pt')
        rw_min=m_reward
        
    
    #clear_output(True)

torch.save(agent.state_dict(), RES_DIR+'/last-model.pt')
tb.close()

env.renderize=True
q_far=np.array([ 0., -0.8 ,  0. , -0.0698,  0.,  3.3825,  0.    ])
NAME_DIR=RES_DIR.split("/")[1]

agent.play(env,NAME_DIR,tmax=800)
