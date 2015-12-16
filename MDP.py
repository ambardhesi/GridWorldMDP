import numpy as np
import argparse
from math import fabs

class GridMDP:
    def __init__(self, reward_matrix, init_state, action_list, terminal_states, gamma = 0.9):
        self.reward_matrix = reward_matrix
        self.init_state = init_state
        self.action_list = action_list
        self.terminal_states = terminal_states
        self.gamma = gamma

    def move(self, state, action):
        """Given a state and an action, return the new state"""
        dimensions = self.reward_matrix.shape
        next = list(state) #convert to list so it is mutable
        if state in self.terminal_states: #Do nothing if we are in a terminal state
            pass
        
        #Check boundary conditions for each action case
        #If action takes us out of boundary, return unchanged variable state
        elif action == "L" and (state[1] > 0 and self.reward_matrix[state[0]][state[1]-1] != None):
            next[1] -= 1
        elif action == "R" and (state[1] < dimensions[1] - 1 and self.reward_matrix[state[0]][state[1]+1] != None):
            next[1] += 1
        elif action == "U" and (state[0] > 0 and self.reward_matrix[state[0] - 1][state[1]] != None):
            next[0] -= 1
        elif action == "D" and (state[0] < dimensions[0] - 1 and self.reward_matrix[state[0] + 1][state[1]] != None):
            next[0] += 1

        return tuple(next)

    def possibleStates(self, state, action):
        """Given a state and action, returns a dict of possible final states along with their probabilities.
        Example : possibleStates((2,3), U) = {(1,3) : 0.8
                                              (2,2) : 0.1
                                              (2,3) : 0.1}
        """
        d = dict()

        if action == 'L':
            if self.move(state, 'L') in d:
                d[self.move(state, 'L')] += 0.8
            else:
                d[self.move(state, 'L')] = 0.8
            if self.move(state, 'U') in d:
                d[self.move(state, 'U')] += 0.1
            else:
                d[self.move(state, 'U')] = 0.1
            if self.move(state, 'D') in d:
                d[self.move(state, 'D')] += 0.1
            else:
                d[self.move(state, 'D')] = 0.1

        elif action == 'R':
            if self.move(state, 'R') in d:
                d[self.move(state, 'R')] += 0.8
            else:
                d[self.move(state, 'R')] = 0.8
            if self.move(state, 'U') in d:
                d[self.move(state, 'U')] += 0.1
            else:
                d[self.move(state, 'U')] = 0.1
            if self.move(state, 'D') in d:
                d[self.move(state, 'D')] += 0.1
            else:
                d[self.move(state, 'D')] = 0.1

        elif action == 'U':
            if self.move(state, 'U') in d:
                d[self.move(state, 'U')] += 0.8
            else:
                d[self.move(state, 'U')] = 0.8
            if self.move(state, 'L') in d:
                d[self.move(state, 'L')] += 0.1
            else:
                d[self.move(state, 'L')] = 0.1
            if self.move(state, 'R') in d:
                d[self.move(state, 'R')] += 0.1
            else:
                d[self.move(state, 'R')] = 0.1

        elif action == 'D':
            if self.move(state, 'D') in d:
                d[self.move(state, 'D')] += 0.8
            else:
                d[self.move(state, 'D')] = 0.8
            if self.move(state, 'L') in d:
                d[self.move(state, 'L')] += 0.1
            else:
                d[self.move(state, 'L')] = 0.1
            if self.move(state, 'R') in d:
                d[self.move(state, 'R')] += 0.1
            else:
                d[self.move(state, 'R')] = 0.1
        
        return d

    def transition(self, state1, action, state2):
        return self.possibleStates(state1, action)[state2]


def value_iteration(MDP, epsilon = 0.001):
    gamma = MDP.gamma
    U1 = np.zeros(MDP.reward_matrix.shape) #initialise all utilities to 0
    PolicyMatrix = np.chararray(MDP.reward_matrix.shape) 
    for terminal_state in MDP.terminal_states: # loop over all terminal states
        U1[terminal_state[0]][terminal_state[1]] = MDP.reward_matrix[terminal_state[0]][terminal_state[1]] # set the end state utilities  
        PolicyMatrix[terminal_state[0]][terminal_state[1]] = 'X'
    
    num_iters = 0
    while True:
        U2 = U1.copy()
        delta = 0
        #print U2
        #print "ITERATION : ", num_iters
        #raw_input("WAITING")

        for i in range(len(MDP.reward_matrix)): 
            for j in range(len(MDP.reward_matrix[i])):   
                state = (i, j)
                if state == (0,3) or state == (1, 3):
                    continue
                if MDP.reward_matrix[i][j] is None:
                    continue
                totalSum = {}
                action_sum = 0
                for action in MDP.action_list:
                    action_sum = 0
                    for possibleState, prob in MDP.possibleStates(state, action).items():      
                        action_sum += prob * U2[possibleState[0]][possibleState[1]]
                    totalSum[action_sum] = action
                    delayed_reward = max(totalSum.keys())
                    policy = totalSum[delayed_reward]
                U1[i][j] = MDP.reward_matrix[i][j] + MDP.gamma * delayed_reward             
                PolicyMatrix[i][j] = policy
            delta = fabs(U2[state[0]][state[1]] - U1[state[0]][state[1]])
        num_iters += 1
        PolicyMatrix[1][1] = 'O'
        for terminal_state in MDP.terminal_states: # loop over all terminal states 
            PolicyMatrix[terminal_state[0]][terminal_state[1]] = 'X'

        if delta <= epsilon * (1 - gamma) / gamma:
            return (U2, PolicyMatrix)

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--reward", required = True)
args = vars(ap.parse_args())

reward = float(args["reward"])

reward_matrix = np.array([[reward, reward, reward, 1],
                 [reward, None, reward, -1],
                 [reward, reward, reward, reward]])

init_state = (2,0)
action_list = ['L', 'R', 'U', 'D']
terminal_states = [(0,3), (1,3)]

MDPProblem = GridMDP(reward_matrix, init_state, action_list, terminal_states, 1)

UtilityMatrix, PolicyMatrix = value_iteration(MDPProblem, 0.001)
print "Utility Matrix : "
print UtilityMatrix
print "Optimal Policies :"
print PolicyMatrix
