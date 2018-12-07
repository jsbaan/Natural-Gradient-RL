from REINFORCE import REINFORCE
from ActorCritic import ActorCritic
from PolicyNetwork import PolicyNetwork
from ValueNetwork import ValueNetwork
import torch
import random
import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from helper import Helper
from NaturalGD import NaturalGD
helper = Helper()
# Create environment
env_name = "CartPole-v0"


def run_reinforce(env_name):
    # Define parameters
    env = gym.envs.make(env_name)
    num_episodes = 200
    num_hidden = 128
    discount_factor = 0.99
    learn_rate = 0.01
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # Initialize policy
    policy = PolicyNetwork(num_hidden)

    # Define optimizer
    optimizer = NaturalGD(policy.parameters(),lr = learn_rate)

    # Perform REINFORCE on environment
    reinforce = REINFORCE(policy, optimizer)
    episode_durations_policy_gradient = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor)

    # Plot loss
    plt.plot(helper.smooth(episode_durations_policy_gradient, 10))
    plt.title('Episode durations per episode')
    plt.legend(['Policy gradient'])
    plt.show()


def run_actorcritic(env_name):
    # Define AC Hyperparamters
    num_hidden = 128
    num_envs = 16
    max_steps = 10000
    max_episodes = 10000
    discount_factor = 0.8
    lr_actor = 1e-3
    lr_critic = 1e-3
    seed = 42
    envs = [gym.envs.make(env_name) for i in range(num_envs)]

    # Initialize networks and model
    actor = PolicyNetwork(num_hidden)
    critic = ValueNetwork(num_hidden)
    actorcritic = ActorCritic(actor, critic, lr_actor,lr_critic)

    for i, env in enumerate(envs):
        env.seed(seed + i)
    torch.manual_seed(seed)

    episode_durations, step_losses = actorcritic.run_episodes(
        envs, max_episodes, max_steps, discount_factor)


# run_actorcritic(env_name)
run_reinforce(env_name)
