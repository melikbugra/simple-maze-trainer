import numpy as np
import gym
import simple_maze_env

class QLearningAgent:

    def __init__(self, info):
        
        self.numActions = info[0]
        self.numStates = info[1]
        self.epsilon = info[2]
        self.learningRate = info[3]
        self.discount = info[4]
        self.randGenerator = np.random.RandomState(info[5])

        self.qTable = np.zeros((self.numStates, self.numActions)) # The array of action-value estimates.

    def agentStart(self, state):
        
        """The method called when the episode starts, it resets the environment.
        
        Args:
            state (int): the state from the
                environment's evnStart function.
        Returns:
            action (int): the first action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        currentQ = self.qTable[state,:] # corresponding row of qTable
        if self.randGenerator.rand() < self.epsilon: # random number generation
            action = self.randGenerator.randint(self.numActions) # take an action randomly
        else:
            action = self.argmax(currentQ) # take an action greedily, the max valued action in currentQ
        self.prevState = state # the trick of TDL
        self.prevAction = action # the trick of TDL
        return action

    def agentStep(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (int): the state from the
                environment's step based on where the agent ended up after the
                last step.
        Returns:
            action (int): the action the agent takes.
        """
        
        # Choose action using epsilon greedy.
        currentQ = self.qTable[state, :] # corresponding row of qTable
        if self.randGenerator.rand() < self.epsilon:  # random number generation
            action = self.randGenerator.randint(self.numActions) # take an action randomly
        else:
            action = self.argmax(currentQ) # take an action greedily, the max valued action in currentQ
        
        self.qTable[self.prevState,self.prevAction] = self.qTable[self.prevState,self.prevAction] + self.learningRate*(reward + self.discount*max(currentQ) - self.qTable[self.prevState,self.prevAction]) # update qTable
        
        self.prevState = state # the trick of TDL
        self.prevAction = action # the trick of TDL
        return action


    def agentEnd(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        
        self.qTable[self.prevState,self.prevAction] = self.qTable[self.prevState,self.prevAction] + self.learningRate*(reward - self.qTable[self.prevState,self.prevAction])

    def argmax(self, qValues):
        """argmax with random tie-breaking
        Args:
            qValues (Numpy array): the array of action-values
        Returns:
            action (int): an action with the highest value
        """
        top = float("-inf") # top as negative infinity
        ties = [] # initiate ties

        for i in range(len(qValues)):
            if qValues[i] > top: # element is greater than top
                top = qValues[i] # the element is the new top
                ties = [] # ties resets

            if qValues[i] == top: # if an element is equal to top
                ties.append(i) # we add its index to ties

        return self.randGenerator.choice(ties) # return one of ties randomly
    
    def agentTest(self, state):
        
        # Choose action greedily by exploiting qTable
        currentQ = self.qTable[state,:]
        
        action = self.argmax(currentQ)
        
        return action

env = gym.make("SimpleMaze-v0")
env.config["sleep"] = 0.0000001
agentInfo = [4, 16, 0.01, 0.1, 0.95, 0] ###numActions, numStates, epsilon, learningRate, discount, random seed
agent = QLearningAgent(agentInfo) # instance of agent
returns = [] # a list that we will append return of every episode
currentReturn = 0 # return of the current episode, we will increment it by rewards

for i in range(100):
    print("Iteration Number: " + str(i))
    currentReturn = 0
    state, rew, done, _ = env.reset()
    act = agent.agentStart(state) # choose action
    currentReturn += rew
    while(True):
        act = agent.agentStep(rew, state) # update q table and choose action
        state, rew, done, _ = env.step(act)
        currentReturn += rew
        if done: # agent is terminated
            agent.agentEnd(rew) # make a final update to q table
            print("Agent is terminated!")
            returns.append(currentReturn) # save the return of the episode
            break

print(agent.qTable)

trainedQTable = agent.qTable

env = gym.make("SimpleMaze-v0")
env.config["sleep"] = 0.0000001
agent = QLearningAgent(agentInfo)
agent.qTable = trainedQTable # we update the agent's q table as trained q table
chosenPath = [] # we create a list for saving the path
returnOfTestEpisode = 0
state, rew, done, _ = env.reset()
chosenPath.append(state)
act = agent.agentTest(state)
state, rew, done, _ = env.step(act)
chosenPath.append(state)
returnOfTestEpisode += rew # we increment test episode return by the reward
while(True):
    act = agent.agentTest(state)
    state, rew, done, _ = env.step(act)
    chosenPath.append(state) # we append the state to chosenPath
    returnOfTestEpisode += rew
    if done:
        print("Agent is terminated!")
        break
print("Test epsiode return: " + str(returnOfTestEpisode))
print(chosenPath)