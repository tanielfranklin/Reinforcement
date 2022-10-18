import numpy as np
from utils import discretize


class DiscreteQLearningAgent:
    def __init__(self, env,
                 # Environment
                 state_grid,  # Discretization state grid
                 # Q-learning alpha parameters:
                 q_alpha=0.02,
                 q_gamma=0.99,
                 # Decayed e-greedy policy parameters:
                 degp_epsilon=1.0,
                 degp_decay_rate=0.9995,
                 degp_min_epsilon=.01
                 ):
        self.env = env
        self.state_grid = state_grid
        # total number of states
        self.state_size = tuple(len(splits) + 1 for splits in self.state_grid)
        # total number of actions
        self.action_size = self.env.action_space.n
        self.q_alpha = q_alpha
        self.q_gamma = q_gamma
        self.degp_epsilon = self.degp_initial_epsilon = degp_epsilon
        self.degp_decay_rate = degp_decay_rate
        self.degp_min_epsilon = degp_min_epsilon
        self.last_state = None
        self.last_action = None
        self.action_history = []
        # Initial Q-map
        self.q = np.zeros(shape=(self.state_size +
                                 (self.action_size,)))
        
    def save_q(self,q_name):
        file_name_table=f"{q_name}.npy"
        np.save(file_name_table,self.q)
        print("q saved")
    def load_q(self,q_name):
        file_name_table=f"{q_name}.npy"
        self.q=np.load(file_name_table)
        print("q loaded")
        

    def transform_state(self, state):
        """State discretization"""
        return tuple(discretize(state, self.state_grid))

    def before_episode(self, state):
        """Adjusting Decayed Epsilon Greedy Policy Parameters before
        new episode"""
        # Decaying epsilon
        self.degp_epsilon *= self.degp_decay_rate
        self.degp_epsilon = max(self.degp_epsilon, self.degp_min_epsilon)
        # Register last state
        self.last_state = self.transform_state(state)
        # Register last actions
        self.last_action = np.argmax(self.q[self.last_state])
        self.action_history = []
        return self.last_action

    def act(self, c_state, reward=None, done=None, mode='train'):
        """Returns actions for the given state"""
        d_state = self.transform_state(c_state)  # Discrete state
        if mode == 'test':
            # greedy decision for 'test' mode
            action = self.greedy_decision(d_state)
        else:
            # Q-learning process for 'train' mode
            last_sa = self.last_state + (self.last_action,)
            # temporal difference
            td = reward + self.q_gamma * max(self.q[d_state]) - self.q[last_sa]
            # update q
            self.q[last_sa] += self.q_alpha * td
            action = self.e_greedy_decision(d_state)
        self.last_state = d_state
        self.last_action = action
        self.action_history.append(action)
        return action

    # classical greedy and e-greedy action
    def e_greedy_decision(self, state):
        """Epsilon Greedy Policy Decision"""
        r = np.random.uniform(0, 1)
        if r < self.degp_epsilon:
            # random choice
            action = np.random.randint(0, self.action_size)
        else:
            # greedy choice
            action = self.greedy_decision(state)
        return action

    def greedy_decision(self, state):
        # greedy only
        action = np.argmax(self.q[state])
        return action

    def direction_changes(self):
        """
        Counting Car direction changes (only 0-Left and 2-Right):
        [0, 1, 2, 1, 0, 0] -> 2
        """
        prev_a = None
        count = 0
        no_ones = [a for a in self.action_history if a != 1]
        for a in no_ones:
            if a != prev_a:
                count += 1
            prev_a = a
        return count

    def run(self, env, episodes, mode='train'):
        """
        Runs episodes:
        - mode = train - Exploration process
        - mode = test - Exploitation process
        """
        total_rewards = []
        d_changes = []
        for e in range(1, episodes + 1):
            state = env.reset()
            action = self.before_episode(state)
            total_reward = 0
            done = False

            while not done:
                state, reward, done, info= env.step(action)
                total_reward += reward
                action = self.act(state, reward, done, mode)
            total_rewards.append(total_reward)
            d_changes.append(self.direction_changes())
            # Printing statistics
            if e % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                avg_dir_changes = np.mean(d_changes[-100:])
                print(
                    f'Episode: {e}/{episodes} | '
                    f'Last 100 Avg TotalReward: {avg_reward} | '
                    f'Last 100 Avg DirChanges: {avg_dir_changes}'
                )
        return total_rewards, d_changes
