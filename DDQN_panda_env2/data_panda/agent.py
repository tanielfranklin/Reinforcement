import numpy as np
import torch.nn as nn
import torch

class DQNAgent(nn.Module):
    def __init__(self, state_shape, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_layer1=128
        self.n_layer2=256
        self.actions_space = self.create_actions_space()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.state_shape = state_shape
        state_dim = state_shape
        # a simple NN with state_dim as input vector (inout is state s)
        # and self.n_actions as output vector of logits of q(s, a)
        self.network = nn.Sequential()
        self.network.add_module('layer1', nn.Linear(state_dim, self.n_layer1))
        self.network.add_module('relu1', nn.ReLU())
        self.network.add_module('layer2', nn.Linear(self.n_layer1, self.n_layer2))
        self.network.add_module('relu2', nn.ReLU())
        self.network.add_module('layer4', nn.Linear(self.n_layer2, self.n_actions))
        self.parameters = self.network.parameters

        self.get_action_index = lambda a: np.where(
            (self.actions_space == a).all(axis=1))[0][0]

    def forward(self, state_t):
        # pass the state at time t through the newrok to get Q(s,a)
        qvalues = self.network(state_t)
        return qvalues

    def create_actions_space(self):
        actions = [-1, 0, 1]
        actions_space = []
        # Create space of actions

        for j1 in actions:
            for j2 in actions:
                for j3 in actions:
                    actions_space.append([j1, j2, j3])
        actions_space.pop(13)  # remove [0,0,0]
        self.n_actions = len(actions_space)
        return actions_space

    def get_qvalues(self, states):
        # input is an array of states in numpy and outout is Qvals as numpy array
        states = np.array(states)
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        # sample actions from a batch of q_values using epsilon greedy policy
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        random_actions = self.actions_space[random_actions[0]]
        best_actions = qvalues.argmax(axis=-1)
        best_actions = self.actions_space[best_actions[0]]
        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    
    def load_weights(self, NAME_DIR, model="best"):
        
        if model == "last":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/last-model.pt'))
        elif model=="best":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model-rw.pt'))
        else:
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model-loss.pt'))

    def play(self, env, NAME_DIR, tmax=500, model="best", q0=[]):
        
        if model == "last":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/last-model.pt'))
        elif model=="best":
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model-rw.pt'))
        else:
            self.load_state_dict(torch.load('runs/'+NAME_DIR+'/best-model-loss.pt'))
            
        if len(q0)!=7:
            s = env.reset()
        else:
            env.panda.q=q0
            s=env.get_state()
        reward = 0
        for step in range(tmax):
            qvalues = self.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0]
            s, r, done, info = env.step(self.actions_space[action])
            reward += r
            if done or info[0] == "terminated":
                break
        print(f'Final score:{reward} in {step} steps')
        print(f"Status: {info[0]} {info[1]}")