
import random
import math
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# NOTE: no batch norm


class LinearFunctionApproximation(object):
    def getQValue(self, phi_a, weightVector):
        qValue = np.dot(weightVector, phi_a)
        return qValue

    def maxQValue(self, weightVector):
        maxq = np.amax(weightVector)
        # For the case of multiple maxima
        j = 0
        M = []
        for i in weightVector:
            if i == maxq:
                M.append(j)
            j += 1

        max_index = random.choice(M)
        return max_index, maxq


# the architecture goes here
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, device):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.n1 = nn.LayerNorm(64)
        self.l2 = nn.Linear(64, 64)
        self.n2 = nn.LayerNorm(64)
        self.l3 = nn.Linear(64, action_dim)

        # device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.max_action = torch.tensor(max_action, dtype=torch.float, device = device )




    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.n1(x)
        x = F.relu(self.l2(x))
        x = self.n2(x)
        x = self.max_action * F.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.n1 = nn.LayerNorm(64)
        self.l2 = nn.Linear(64 + action_dim, 64)
        self.n2 = nn.LayerNorm(64)
        self.l3 = nn.Linear(64, 1)


    def forward(self, x, u):
        x = F.relu(self.l1(x))
        x = self.n1(x)
        x = F.relu(self.l2(torch.cat([x, u], 1)))
        x = self.n2(x)
        x = self.l3(x)
        return x



class LinearActor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device):

        super(LinearActor, self).__init__()

        # only one linear layer
        self.l1 = nn.Linear(state_dim, action_dim)

        # move the action cliping in the agent itself

    def forward(self, x):
        return self.l1(x)



class LinearCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(LinearCritic, self).__init__()

        # just a linear layer on the concatenation of features
        self.l1 = nn.Linear(state_dim + action_dim, 1)

    def forward(self, x, u):
        return self.l1(torch.cat([x, u], 1))
