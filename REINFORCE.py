import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm

class REINFORCE():
    def __init__(self,policy,learn_rate):
        self.policy = policy
        self.learn_rate = learn_rate

    def select_action(self, state):
        # Samples an action according to the probability distribution induced by the model
        # Also returns the log_probability
        log_p = self.policy.forward(torch.Tensor(state).unsqueeze(0))[0]
        action = torch.multinomial(log_p.exp(),1).item()
        return action, log_p[action]

    def run_episode(self, env, discount_factor):
        s = env.reset()
        done = False
        episode = []
        r_list = []
        G = 0
        steps = 0
        while not done:
            a, log_p = self.select_action(s)
            s_next, r, done, _ = env.step(a)
            episode.append((s,a,log_p,r,s_next))
            s = s_next
            G += r * discount_factor**steps
            steps += 1
        return episode, G

    def compute_reinforce_loss(self,episode, discount_factor):
        returns = torch.zeros(len(episode))
        a_probs = torch.zeros(len(episode))

        for i,(s,a,log_p,r,s_next) in enumerate(reversed(episode)):
            if i == 0:
                returns[i] = r
            else:
                returns[i] = discount_factor * returns[i-1] + r
            a_probs[i] = log_p

        # Normalize returns
        returns = (returns - returns.mean())/returns.std()
        loss = - torch.sum(returns * a_probs)
        return loss

    def run_episodes_policy_gradient(self, env, num_episodes, discount_factor, optimizer):

        episode_returns = []
        losses = []
        total_loss = 0
        for i in range(num_episodes):
            # Run episode
            episode, G = self.run_episode(env, discount_factor)

            # Compute loss
            loss = self.compute_reinforce_loss(episode, discount_factor)

            # total_loss += loss
            # if i % 10 ==0:
            #     print(total_loss/(i+1))

            # Train network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print("Episode {0} finished with {1} return"
                      .format(i, G))
            episode_returns.append(G)
            losses.append(loss)
        return episode_returns, losses
