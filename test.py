import gym
import simple_maze_env
import random
from simple_maze_dqn import DQNAgent
from collections import deque
import torch as T
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import random
import pickle

device = T.device('cpu')
env = gym.make("SimpleMaze-v0")
env.config["sleep"] = 1


agent = DQNAgent(env, 10)
agent.model.load_state_dict(T.load("agent.pth"))
agent.epsilon = 0
for i in range(10):
    done = False
    state = env.reset()
    state = np.reshape(state, [1,2])
    state = T.from_numpy(state)
    state = Variable(state).to(device)
    env.render(mode="graphics")
    while not done:
        act = agent.act(state)
        print(act)
        state, rew, done, _ = env.step(act)
        print(rew)
        state = np.reshape(state, [1,2])
        state = T.from_numpy(state)
        state = Variable(state).to(device)
        env.render(mode="graphics")
        a=input()