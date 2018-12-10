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
from SGD_NG import SGD_NG
from RMSProp_NG import RMSProp_NG
from torch.optim import RMSprop
helper = Helper()
# Create environment
env_name = "CartPole-v0"


def run_reinforce(env_name):
    # Define parameters
    env = gym.envs.make(env_name)
    num_actions = env.action_space.n
    num_states = env.reset().shape[0]
    num_episodes = 100
    num_hidden = 64
    discount_factor = 0.99
    learn_rate_ng = 0.01
    learn_rate_vg = 0.001
    learn_rate_adam = 0.01
    learn_rate_rms = 0.005
    learn_rate_rmsng = 0.001


    seed = 45
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    print("Number of states: ", num_states)
    print("Number of actions: ", num_actions)

    ## Natural GD
    model = PolicyNetwork(num_states, num_actions, num_hidden)
    optimizer_NG = SGD_NG(model.parameters(), learn_rate_ng)
    reinforce = REINFORCE(model,optimizer_NG)
    episode_returns_policy_gradient_NG = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor)

    ## Vanilla SGD
    model = PolicyNetwork(num_states, num_actions, num_hidden)
    optimizer_VG = torch.optim.SGD(model.parameters(), learn_rate_vg)
    reinforce = REINFORCE(model,optimizer_VG)
    episode_returns_policy_gradient_VG = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor)

    ## Adam
    print('Adam...')
    model = PolicyNetwork(num_states, num_actions, num_hidden)
    optimizer_ADAM = torch.optim.Adam(model.parameters(), learn_rate_adam)
    reinforce = REINFORCE(model,optimizer_ADAM)
    episode_returns_policy_gradient_adam = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor)

    ## RMSProp Vanilla
    print('RMSProp Vanilla...')
    model = PolicyNetwork(num_states, num_actions, num_hidden)
    optimizer_RMS = RMSprop(model.parameters(), learn_rate_rms)
    reinforce = REINFORCE(model,optimizer_RMS)
    episode_returns_policy_gradient_rms = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor)

    ## RMSProp NG
    print('RMSProp NG...')
    model = PolicyNetwork(num_states, num_actions, num_hidden)
    optimizer_RMSNG = RMSProp_NG(model.parameters(), learn_rate_rmsng,ng=True)
    reinforce = REINFORCE(model,optimizer_RMSNG)
    episode_returns_policy_gradient_rmsng = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor)

    # Plot loss
    plt.plot(helper.smooth(episode_returns_policy_gradient_NG, 10))
    plt.plot(helper.smooth(episode_returns_policy_gradient_VG, 10))
    plt.plot(helper.smooth(episode_returns_policy_gradient_adam, 10))
    plt.plot(helper.smooth(episode_returns_policy_gradient_rms, 10))
    plt.plot(helper.smooth(episode_returns_policy_gradient_rmsng, 10))

    plt.title('Episode returns per episode')
    # plt.legend(['Adam','RMS','RMSNG'])

    plt.legend(['Natural Policy gradient (SGD)', 'Vanilla Policy Gradient (SGD)', 'Adam','RMS','RMSNG'])
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
