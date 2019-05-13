import gym
import numpy as np
import torch
from datetime import datetime
import sys


from collections import namedtuple
#from envs.double_pendulum_env_x import DoublePendulumEnvX
#from envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
#from envs.half_cheetah_env_x import HalfCheetahEnvX, HalfCheetahEnvXLow1, HalfCheetahEnvXLow2, HalfCheetahEnvXHigh1
#from rllab.misc import ext
#from envs.sparse_mountain_car import SparseMountainCarEnv

#from envs.maze.point_maze_env import OneRoomPointMazeEnv, CustomPointMazeEnv

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state',
                                       'done',))


CUSTOM_RLLAB_ENVIRONMENTS = ["DoublePendulumEnvX",
                       "CartpoleSwingupEnvX",
                       "HalfCheetahEnvX",
                       "HalfCheetahEnvXLow1",
                       "HalfCheetahEnvXLow2",
                       "HalfCheetahEnvXHigh1",
                       "OneRoomPointMazeEnv",
                       ]

CUSTOM_MAZE_ENVIRONMENTS = ["CustomPointMazeEnv",]


CUSTOM_GYM_ENVIRONMENTS = ["SparseMountainCarEnv", ]


class Memory(object):
    def __init__(self,
                 max_size = 5000):
        """
        can optimize further by using a numpy array and allocating it to zero
        """
        self.max_size = max_size
        self.store = [None] * self.max_size  # is a list, other possible data structures might be a queue
        self.count = 0
        self.current = 0


    def add(self, transition):
        """ insert one sample at a time """

        self.store[self.current] = transition

        # for taking care of how many total transitions have been inserted into the memory
        self.count = max(self.count, self.current + 1)

        # increase the counter
        self.current = (self.current + 1) % self.max_size

    def get_sample(self, index):
        # normalize index
        index = index % self.count

        return self.store[index]

    def get_minibatch(self, batch_size = 100):
        """
        a minibatch of random transitions without repetition
        """
        ind = np.random.randint(0, self.count, size=batch_size)
        samples = []

        for index in ind:
            samples.append(self.get_sample(index))

        return samples


def create_env(env_name = 'Swimmer-v1', init_seed=0):
    """
    create a copy of the environment and set seed
    """
    env = gym.make(env_name)
    env.seed(init_seed)

    return env


def log(msg):
    print("[%s]\t%s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg))
    sys.stdout.flush()


def soft_update(target, source, tau):
    """
    do the soft parameter update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def create_rllab_env(env_name, init_seed):
    """
    create the rllab env
    """
    env = eval(env_name)()
    ext.set_seed(init_seed)
    return env


def create_gym_env(env_name, init_seed):
    """
    create the rllab env
    """
    env = eval(env_name)()
    env.seed(init_seed)

    return env


def create_maze_env(env_name, init_seed, maze_id, maze_length):
    """
    create the rllab env
    """
    env = eval(env_name)(maze_id, maze_length)
    ext.set_seed(init_seed)
    return env
