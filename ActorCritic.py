import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
from helper import Helper

class ActorCritic():
    def __init__(self, actor, critic, actor_learn_rate, critic_learn_rate):
        self.actor = actor
        self.critic = critic
        self.actor_learn_rate = actor_learn_rate
        self.critic_learn_rate = critic_learn_rate
        return

    def select_action(self,state):
        # Samples an action according to the probability distribution induced by the model
        # Also returns the log_probability
        log_p = self.actor.forward(torch.Tensor(state))
        action = torch.multinomial(log_p.exp(),1)
        log_p = log_p.gather(1,action).squeeze()
        action = action.squeeze(1)

        # action and log_p should be a 1 dimensional vector
        n = len(state)
        assert action.size() == (n, )
        assert log_p.size() == (n, )
        return action, log_p

    def train_actor_critic(self,optimizer, log_ps, state, reward, next_state, done, discount_factor):
        # Compute value/critic loss
        cur_value = self.critic(state).squeeze(1)
        with torch.no_grad():
            mask = (done==False).float()
            next_value = mask*(discount_factor * self.critic(next_state).squeeze(1)) + reward
        value_loss = F.smooth_l1_loss(cur_value, next_value)

        # Compute actor loss
        v = reward + discount_factor * self.critic(next_state).squeeze(1)
        actor_loss = -torch.mean(log_ps * v.detach())


        # The loss is composed of the value_loss (for the critic) and the actor_loss
        loss = value_loss + actor_loss

        # backpropagation of loss to Neural Network (PyTorch magic)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item(), value_loss.item(), actor_loss.item()  # Returns a Python scalar, and releases history (similar to .detach())

    def run_episodes(self,envs, max_episodes, max_steps, discount_factor):

        # We can use a single optimizer for both the actor and the critic, even with separate learn rates
        optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.actor_learn_rate},
            {'params': self.critic.parameters(), 'lr': self.critic_learn_rate}
        ])

        episode_durations = []
        state = torch.tensor([env.reset() for env in envs], dtype=torch.float)
        current_episode_lengths = torch.zeros(len(envs), dtype=torch.int64)
        step_losses = []  # Keep track of losses for plotting
        for i in range(max_steps):

            if i % 100 == 0:
                print(f"Step {i}, finished {len(episode_durations)} / {max_episodes} episodes, average episode duration of last 100 episodes: {np.mean(episode_durations[-100:])}")

            action, log_ps = self.select_action(state)
            next_state, reward, done, _ = zip(*[env.step(a.item()) for env, a in zip(envs, action)])

            next_state = torch.tensor(next_state, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)
            done = torch.tensor(done, dtype=torch.uint8)  # Boolean
            current_episode_lengths += 1

            losses = self.train_actor_critic(optimizer, log_ps, state, reward, next_state, done, discount_factor)

            step_losses.append(losses)

            # Reset envs that are done
            next_state = torch.tensor([
                env.reset() if d else s.tolist()
                for env, s, d in zip(envs, next_state, done)
            ], dtype=torch.float)

            episode_durations.extend(current_episode_lengths[done])
            current_episode_lengths[done] = 0  # PyTorch can also work in place

            state = next_state

            # Check if we have finished sufficiently many episodes
            if len(episode_durations) >= max_episodes:
                break

        return episode_durations[:max_episodes], step_losses  # In case we want exactly num_episodes returned
