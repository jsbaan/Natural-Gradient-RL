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
helper = Helper()
# Create environment
env_name = "CartPole-v0"


def run_reinforce(env_name):
    # Define parameters
    env = gym.envs.make(env_name)
    num_actions = env.action_space.n
    num_states = env.reset().shape[0]
    num_episodes = 300
    num_hidden = 128
    discount_factor = 0.99
    learn_rate = 0.01
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    print("Number of states: ", num_states)
    print("Number of actions: ", num_actions)

    # Initialize policy
    model = PolicyNetwork(num_states, num_actions, num_hidden)
    reinforce = REINFORCE(model,learn_rate)

    optim_list = {
        "SGD_NG": SGD_NG(model.parameters(), learn_rate),
        "SGD": torch.optim.SGD(model.parameters(), learn_rate),
        "ADAM": torch.optim.Adam(model.parameters(), learn_rate),
    }

    episode_return_list = {}

    for optim_name, optim in optim_list.items():
        print('--- %s ---' %optim_name)
        episode_returns = reinforce.run_episodes_policy_gradient(
            env, num_episodes, discount_factor, optim)

        episode_return_list[optim_name] = episode_returns

    # Plot loss
    for optim_name, episode_returns in episode_return_list.items():
        plt.plot(helper.smooth(episode_returns, 10), label=optim_name)
        plt.title('Episode returns per episode')
        plt.legend()
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
