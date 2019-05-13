import math
from PIL import Image
import numpy as np
import os
import time
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy


from tensorboardX import SummaryWriter

from utils import *
from exploration_models import *
# from models import LinearActor, LinearCritic
from models import LinearFunctionApproximation
import csv


class LinearAgent(object):
    """
    The DDPG Agent
    """
    def __init__(self,
                 args,
                 env,
                 exploration,
                 featurize_state,
                 featurize_action,
                 writer_dir = None,
                 csv_dir = None):
        """
        init agent
        """
        self.env = env
        self.args = args
        self.exploration = exploration
        self.featurize_state = featurize_state
        self.featurize_action = featurize_action

        self.state_feature_dim = self.args.state_num_features
        self.action_feature_dim = self.args.action_num_features
        self.action_dim = env.action_space.shape[0]
        self.max_action = float(env.action_space.high[0])

        self.device = torch.device("cuda" if (torch.cuda.is_available() and  self.args.gpu) else "cpu")

        self.qlearner = LinearFunctionApproximation()
        # TODO: set the random seed in the main launcher
        # set the seeds here again
        # Susan removed the following lines
        #random.seed(self.args.seed)
        #torch.manual_seed(self.args.seed)
        #np.random.seed(self.args.seed)
        if self.args.gpu:
            torch.cuda.manual_seed(self.args.seed )


        # create the models and target networks
#        self.actor = LinearActor(self.state_dim, self.action_dim, self.max_action, self.device).to(self.device)
#        self.actor_target = LinearActor(self.state_dim, self.action_dim, self.max_action, self.device).to(self.device)
#        self.actor_target.load_state_dict(self.actor.state_dict())
#        # TODO: use adam??
#        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr)
#
#        self.critic = LinearCritic(self.state_dim, self.action_dim).to(self.device)
#        self.critic_target = LinearCritic(self.state_dim, self.action_dim).to(self.device)
#        self.critic_target.load_state_dict(self.critic.state_dict())
#
#        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=  10*self.args.lr, weight_decay=1e-2)

        # use memory or not?
        self.memory = Memory(self.args.replay_size)


        self.writer = SummaryWriter(log_dir=writer_dir) if  writer_dir is not None else None

        self.total_steps = 0 
        self.iteration = 0 # Susan added this line
        self.episode_num = 0 # Susan added this line
        self.exp_flag = 1 # Susan added this line
        self.eval_flag = 0 # Susan added this line
        self.num_goal_reached =0 # Susan added this line


        #Susan added the following lines
        self.csv_dir = csv_dir
        row_Ext = ['ITER_NUM', 'EP_N', 'EVAL_N', 'REWARD', 'EVAL_STEPS']
        row_Con = ['ITER_NUM', 'EP_N', 'AVG_REW', 'AVG_STEPS', 'REW_ERR', 'STEPS_ERR']
        #row_Sum = ['EP_N', 'TOT_AVG_REW', 'TOT_AVG_STEPS', 'TOT_REW_ERR', 'TOT_STEPS_ERR']
        with open(os.path.join(self.csv_dir,'Eval_Report_Extensive.csv'), 'w') as csvFile_Ext:
            writer = csv.writer(csvFile_Ext)
            writer.writerow(row_Ext)
        csvFile_Ext.close()
        with open(os.path.join(self.csv_dir,'Eval_Report_Concise.csv'),'w') as csvFile_Con:
            writer = csv.writer(csvFile_Con)
            writer.writerow(row_Con)
        csvFile_Con.close()
        #with open(os.path.join(self.csv_dir,'Eval_Report_Summary.csv'), 'w') as csvFile_Sum:
        #    writer = csv.writer(csvFile_Sum)
        #    writer.writerow(row_Sum)
        #csvFile_Sum.close()


    def actionRegionAngleBound(self):
        min_range = self.max_index * 2 * math.pi/self.args.action_num_features
        max_range = min_range + (2 * math.pi/self.args.action_num_features)
        return min_range, max_range
    def resetActionMatrix(self):
        self.actionMatrix = np.zeros((self.args.state_num_features, self.args.action_num_features))
        return self.actionMatrix
        
    def setWeightVector(self, state):
        state_index = self.featurize_state(state)
        weightVector = copy.copy(self.actionMatrix[state_index,])
        self.weightVector = weightVector
        return weightVector

    def pi(self, state):
        """
        take the action based on the current policy
        """
        max_index, maxq = self.qlearner.maxQValue(self.weightVector)

        self.max_index = max_index
        self.maxq = maxq

        min_range, max_range = self.actionRegionAngleBound()
        theta = random.uniform(min_range, max_range)
        
        while not (min_range <= theta < max_range):
            theta = random.uniform (min_range, max_range)
            # choose a point on the line passing through origin with slope angle theta
        if 0 <= theta < math.pi / 2 or 3 * math.pi / 2 <= theta < math.pi * 2:
            x = random.uniform(0 ,+ self.max_action)
        else:
            x = random.uniform(- self.max_action, 0)
        y = x * math.tan(theta)
        
        action = [x, y]
        return action


    def exp_pi(self, state):
        """
        gives the noisy action
        """
        # get action according to the policy
        action = self.pi(state)

        if self.exploration is not None:
            prev_action = action
            if isinstance(self.exploration, PolyNoise):
                raise Exception("not implemented for exploration only agent")
            elif isinstance(self.exploration, GyroPolyNoise):
                raise Exception("not implemented for exploration only agent")
            elif isinstance(self.exploration, GyroPolyNoiseActionTraj):
                action, exp_flag = self.exploration(action, state)
                self.exp_flag = exp_flag

                # log every 1k steps
                if self.total_steps % 250 == 0 and self.writer is not None:
                    # this should be increasing throughout the episode
                    self.writer.add_scalar("radius_g", self.exploration.g, self.total_steps/1000)
            elif isinstance(self.exploration, PolyNoiseTrajEpsilon):
                action, exp_flag = self.exploration(action, state, self.num_goal_reached)

                self.exp_flag = exp_flag

            elif isinstance(self.exploration, OrnsteinUhlenbeckActionNoise):
                # OU Noise
                noise = self.exploration()
                assert noise.shape == action.shape
                action += noise
            else:
                # random noise
                # uniform sampling from the action space
                action = self.env.action_space.sample()

        # clip the action
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action


    def compute_td_error(self, transition):
        """
        take the transition and return the td_error for that
        """
        batch = transition
        bs = 1

        with torch.no_grad():
            state = torch.from_numpy(np.asarray(batch.state).reshape(bs, -1)).float().to(self.device)
            action = torch.from_numpy(np.asarray(batch.action).reshape(bs, -1)).float().to(self.device)
            reward = torch.from_numpy(np.asarray(batch.reward).reshape(bs, -1)).float().to(self.device)
            next_state = torch.from_numpy(np.asarray(batch.next_state).reshape(bs, -1)).float().to(self.device)
            done = torch.from_numpy(np.asarray(batch.done).reshape(bs, -1)).float().to(self.device)

            Q_value = self.critic(state, action)
            Target_Q_estimate = self.critic_target(next_state, self.actor_target(next_state))
            Q_target_value = (reward + ( (1. - done) * self.args.gamma * Target_Q_estimate)).detach()

            td_error = (Q_target_value - Q_value)

        return td_error.item()



    def Q_update(self, state, action, next_state, reward):
        action_index, phi_a = self.featurize_action(action)
        # an array of Q(s,a) for all possible a
        current_weightVector = self.setWeightVector(state)
        # an array of Q(s',a') for all possible a'
        next_weightVector = self.setWeightVector(next_state)
        
        Q_value = self.qlearner.getQValue(phi_a, current_weightVector)
        _, Target_Q_estimate = self.qlearner.maxQValue(next_weightVector)
        Q_target_value = reward + self.args.gamma * Target_Q_estimate
        td_error = Q_target_value - Q_value
        
        current_weightVector[action_index] = Q_value + self.args.lr * td_error

        state_index = self.featurize_state(state)

        self.actionMatrix[state_index,] = current_weightVector
        

    def off_policy_update(self, epoch):
        """
        the update to network
        """
        actor_loss_list = []
        critic_loss_list = []
        pred_v_list = []

        for _ in range(self.args.updates_per_step):

            # sample a transition buffer
            transitions = self.memory.get_minibatch(self.args.batch_size)
            batch = Transition(*zip(*transitions))
            bs = self.args.batch_size

            state = torch.from_numpy(np.asarray(batch.state).reshape(bs, -1)).float().to(self.device)
            action = torch.from_numpy(np.asarray(batch.action).reshape(bs, -1)).float().to(self.device)
            reward = torch.from_numpy(np.asarray(batch.reward).reshape(bs, -1)).float().to(self.device)
            next_state = torch.from_numpy(np.asarray(batch.next_state).reshape(bs, -1)).float().to(self.device)
            done = torch.from_numpy(np.asarray(batch.done).reshape(bs, -1)).float().to(self.device)

            Q_value = self.critic(state, action)
            Target_Q_estimate = self.critic_target(next_state, self.actor_target(next_state))
            Q_target_value = (reward + ( (1. - done) * self.args.gamma * Target_Q_estimate)).detach()

            td_error = Q_target_value - Q_value

            # critic loss
            critic_loss = F.mse_loss(Q_value, Q_target_value)

            # optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute the actor loss
            actor_loss = - self.critic(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            soft_update(self.actor_target, self.actor, self.args.tau)
            soft_update(self.critic_target, self.critic, self.args.tau)


            # append to the list
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
            pred_v_list.append(Q_value.mean().item())

            # TODO: can log the loss here
            # log every 10 steps
            if self.total_steps % 1000 == 0:
                self.writer.add_scalar("actor_loss", np.mean(actor_loss_list), epoch)
                self.writer.add_scalar("critic_loss", np.mean(critic_loss_list), epoch)
                self.writer.add_scalar("pred_v", np.mean(pred_v_list), epoch)

        return [np.mean(actor_loss_list), np.mean(critic_loss_list), np.mean(pred_v_list)]

    def run(self):
        """
        the actual ddpg algorithm here
        """
        results_dict = {
            "train_rewards" : [],
            "eval_rewards" : [],
            "ep_len" : [],
            "g_history" : [],
            "step_train_rewards" : [],
        }

        update_steps = 0
        eval_steps = 0 

        sum_train_rewards = 0

        self.total_steps = 0 
        # num_episodes = 0 # (Susan: Removed in the new version)

        # reset state and env
        # reset exploration porcess
        state = self.env.reset()

        # self.actionMatrix = self.resetActionMatrix()
        # self.weightVector = self.setWeightVector(state)
        # featurize the state here
        # state = self.featurize_state(state)

        done = False
        if self.exploration is not None:
            self.exploration.reset()

        #ep_reward = 0
        #ep_len = 0
        start_time = time.time()

        #The following lines are for visual purposes
        if self.args.Graph:
            traj=[]
            imp_states=[]
            traj.append(state)


        # Susan: modified the following for loop in the new version
        for iteration in range(self.args.num_iter):
            self.env.num_goal_reached = 0
            self.iteration = iteration
            self.actionMatrix = self.resetActionMatrix()

            for ep in range(self.args.num_ep):
                self.episode_num = ep
    
                # reset state and env
                # reset exploration porcess
                state = self.env.reset()

                self.weightVector = self.setWeightVector(state)
    
                # featurize the state here
                # state = self.featurize_state(state)
    
                done = False
                if self.exploration is not None:
                    self.exploration.reset()
    
                ep_reward = 0
                ep_len = 0
    
                while not done:
    
                    # convert the state to tensor
                    # state_tensor = torch.from_numpy(state).float().to(self.device).view(-1, self.state_dim)
        
                    # get the expl action
                    action = self.exp_pi(state)
                    #print(action)
    
        
                    if self.args.Graph:
                        win1, next_state, reward, done, num_goal_reached, _ = self.env.step(action, self.exp_flag, self.eval_flag)
                        traj.append(next_state)
                        if (ep_len % 100 == 0 and ep_len != 0) or done == True:
                            imp_states.append(next_state)
                    else:
                        next_state, reward, done, num_goal_reached, _ = self.env.step(action, self.exp_flag, self.eval_flag)
                    # next_state = self.featurize_state(next_state)
                    self.num_goal_reached = num_goal_reached # Susan added this line
                    ep_reward += reward
                    ep_len += 1 
                    self.total_steps += 1
        
                    # hard reset done for rllab envs
                    # done = done or ep_len >= self.args.max_path_len # (Susan: Removed in the new version)
        
                    # add the transition in the memory
                    transition = Transition(state = state, action = action,
                                               reward = reward, next_state = next_state,
                                               done = float(done))
        
                    self.memory.add(transition)
        
                    # update  
                    self.Q_update(state, action, next_state, reward) 
                    # update the state
                    state = next_state
        
        
                    # update here
                    # if self.memory.count > self.args.batch_size * 5:
                     #   self.off_policy_update(update_steps)
                     #   update_steps += 1

        
        
                    # if self.total_steps % self.args.checkpoint_interval == 1:
                     #   self.save_models()
                     #   torch.save(results_dict, os.path.join(self.args.out, 'results_dict.pt'))
        
        
                # log
                results_dict["train_rewards"].append(ep_reward)
                results_dict["ep_len"].append(ep_len)
        
                sum_train_rewards += ep_reward
        
                self.writer.add_scalar("ep_reward", ep_reward, ep+1)
                self.writer.add_scalar("ep_len", ep_len, ep+1)
                self.writer.add_scalar("reward_step", ep_reward, self.total_steps)
                self.writer.add_scalar("avg_ep_len", np.mean(results_dict["ep_len"][-100:]), ep+1)
        
                log(
                    'Num Episode {}\t'.format(ep+1) + \
                    'Time: {:.2f}\t'.format(time.time() - start_time) + \
                    'E[R]: {:.2f}\t'.format(ep_reward) +\
                    'E[t]: {}\t'.format(ep_len) +\
                    'Step: {}\t'.format(self.total_steps) +\
                    'Epoch: {}\t'.format(self.total_steps // 10000) +\
                    'avg_search_time: {:.2f}\t'.format(np.mean(results_dict["ep_len"][-100:])) +\
                    'avg_train_reward: {:.2f}\t'.format(np.mean(results_dict["train_rewards"][-100:]))
                    )
        
        
                if self.exploration is not None and (
                        isinstance(self.exploration, GyroPolyNoiseActionTraj) ):
        
                    # get the radius of gyration per episode
                    g_history = np.asarray(self.exploration.g_history)
                    results_dict["g_history"].append(g_history)
        
        
                # reset
                state = self.env.reset()
                state = self.featurize_state(state)
                done = False
                if self.exploration is not None:
                    self.exploration.reset()
                ep_reward = 0
                ep_len = 0
                start_time = time.time()
        
                # update counters
                # num_episodes += 1 # (Susan: Removed in the new version)
        
                """ 
                eval the policy here after an episode 
                """ 
                eval_ep, eval_len, eval_ep_err, eval_len_err = self.eval()
                results_dict["eval_rewards"].append(eval_ep)
    
                log('----------------------------------------')
                log('Eval[R]: {:.2f}\t'.format(eval_ep) +\
                    'Eval[t]: {}\t'.format(eval_len) +\
                    'avg_eval_reward: {:.2f}\t'.format(np.mean(results_dict["eval_rewards"][-10:]))
                    )
                log('----------------------------------------')
    
                self.writer.add_scalar("eval_reward", eval_ep, eval_steps) # Susan: Removed in the new version
    
                eval_steps += 1 # Susan: Removed in the new verison
    
    
                # also log the cum train rewards
                self.writer.add_scalar("cum_train_rewards", sum_train_rewards, eval_steps)
                # add to dict also
                results_dict["step_train_rewards"].append(sum_train_rewards)
    
    
                #Susan added the following block in the new version 
                row_Con = [self.iteration+1, self.episode_num+1, eval_ep, eval_len, eval_ep_err, eval_len_err]
                with open(os.path.join(self.csv_dir,'Eval_Report_Concise.csv'),'a') as csvFile_Con:
                    writer = csv.writer(csvFile_Con)
                    writer.writerow(row_Con)
                csvFile_Con.close()

        # done with all the training

        # save the models
        # self.save_models()

        # save the results
        torch.save(results_dict, os.path.join(self.args.out, 'results_dict.pt'))

        # Open and save the graph
        if self.args.Graph:
            #img=self.env.vis_trajectory(np.asarray(traj),np.asarray(imp_states))
            #im= Image.open(img)
            #im.show()
            win1.getMouse()
            win1.postscript(file="image.eps",colormode='color')
            win1.close()



    def eval(self):
        """
        evaluate the current policy and log it
        """
        print(self.actionMatrix)
        avg_reward = []
        avg_len = []
        self.exp_flag = 0 # Susan added this line
        self.eval_flag = 1 # Susan added this line

        for eval_i in range(self.args.eval_n):

            state = self.env.reset()
            # state = self.featurize_state(state)
            done = False

            ep_reward = 0
            ep_len = 0
            start_time = time.time()

            while not done:

                # convert the state to tensor
        #        state_tensor = torch.from_numpy(state).float().to(self.device).view(-1, self.state_dim)

                weightVector = self.setWeightVector(state)
                self.weightVector = weightVector
                # get the policy action
                action = self.pi(state)

                #Susan added this line to avoid errors
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                if self.args.Graph:
                    win1, next_state, reward, done, num_goal_reached, _ = self.env.step(action, self.exp_flag, self.eval_flag)
                else:
                    next_state, reward, done, num_goal_reached, _ = self.env.step(action, self.exp_flag, self.eval_flag)
                # next_state = self.featurize_state(next_state)
                self.num_goal_reached = num_goal_reached # Susan added this line
                ep_reward += reward
                ep_len += 1

                # update the state
                state = next_state

                done = done or ep_len >= self.args.eval_max_path_len
                
            
            # Susan: added the following block in the new version
            row_Ext = [self.iteration+1, self.episode_num+1, eval_i+1, ep_reward, ep_len]
            with open(os.path.join(self.csv_dir,'Eval_Report_Extensive.csv'), 'a') as csvFile_Ext:
                writer = csv.writer(csvFile_Ext)
                writer.writerow(row_Ext)
            csvFile_Ext.close()


            avg_reward.append(ep_reward)
            avg_len.append(ep_len)
            #print('state in eval:',state)
        self.eval_flag = 0 # susan added this line
        return np.mean(avg_reward), np.mean(avg_len), stats.sem(avg_reward), stats.sem(avg_len)



    def save_models(self):
        """create results dict and save"""
        models = {
        "actor" : self.actor.state_dict(),
        "critic" : self.critic.state_dict(),
        }
        torch.save(models,os.path.join(self.args.out, 'models.pt'))


    def load_models(self):
        models = torch.load(os.path.join(self.args.out, 'models.pt'))
        self.actor.load_state_dict(models["actor"])
        self.critic.load_state_dict(models["critic"])
