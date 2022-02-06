from collections import deque
import torch as T
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import random
import gym
import simple_maze_env
import pickle


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, fc1_dim, fc2_dim, out_dim, lr):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, out_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class DQNAgent:
    def __init__(self, env, episodes):
        self.env = env
        

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.gamma = 0.95
        self.learning_rate = 0.01

        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon - self.epsilon_min)/max(1, int(episodes/10))

        self.memory = deque(maxlen=1000)

        self.model = NeuralNetwork(self.state_size, 64, 64, self.action_size, self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        # storage
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.uniform(0, 1) <= self.epsilon:
            # print("random")
            return random.randint(0,3)
        else:
            act_values = self.model(state).cpu().detach().numpy()
            # print(state)
            # print(act_values)
            # print(np.argmax(act_values))
            return np.argmax(act_values)

    def replay(self, batch_size):
        # training
        if len(self.memory) < batch_size:
            return
        mini_bacth = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_bacth:
            if done:
                target = reward
            else:
                target = reward + self.gamma*np.amax(self.model(next_state).cpu().detach().numpy())
            # print(state)
            train_target = self.model(state)
            # print(train_target)
            train_target[0][action] = target

            self.model.optimizer.zero_grad()
            Q_values = self.model(state)
            mse = nn.MSELoss()
            loss = mse(Q_values, train_target)
            loss.backward()
            self.model.optimizer.step()
    
    def adaptive_e_greedy(self):
        if self.epsilon > self.epsilon_min + 0.0000001:
            self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

if __name__ == "__main__":
    device = T.device('cpu')
    # initialize environment
    env = gym.make("SimpleMaze-v0")
    env.config["sleep"] = 0.00000000000001
    episodes = 10000
    agent = DQNAgent(env, episodes)
    
    batch_size = 16
    for e in range(episodes):

        state= env.reset()
        state = np.reshape(state, [1,2])
        state = T.from_numpy(state)
        state = Variable(state).to(device)
    

        score = 0
        while True:
            env.render(mode='graphics')
            # act
            action = agent.act(state)

            # step
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1,2])
            next_state = T.from_numpy(next_state)
            next_state = Variable(next_state).to(device)

            # remember
            agent.remember(state, action, reward, next_state, done)

            # print(state, action, reward, next_state, done)

            # update state
            state = next_state

            # replay
            agent.replay(batch_size)

            # adjust epsilon
            agent.adaptive_e_greedy()
            # print(agent.epsilon)

            score += reward

            # a = input()


            if done:
                print("\n")
                print(f"Episode: {e}, Score: {score}, Epsilon: {agent.epsilon}")
                break

    T.save(agent.model.state_dict(), "agent.pth")