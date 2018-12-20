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
from SGD_NG import SGD
helper = Helper()
# Create environment
env_name = "CartPole-v0"
NUMBER_OF_RUNS = 2

def run_reinforce(env_name):
    # Define parameters
    env = gym.envs.make(env_name)
    num_actions = env.action_space.n
    num_states = env.reset().shape[0]
    num_episodes = 300
    num_hidden = 128
    discount_factor = 0.99
    learn_rate = 0.03
    learn_rate_VG = 0.01
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    print("Number of states: ", num_states)
    print("Number of actions: ", num_actions)

    ## Save stats
    SGD_results = np.zeros((NUMBER_OF_RUNS, num_episodes))
    NG_results = np.zeros((NUMBER_OF_RUNS, num_episodes))
    SGD_loss = np.zeros((NUMBER_OF_RUNS, num_episodes))
    NG_loss = np.zeros((NUMBER_OF_RUNS, num_episodes))

    # Initialize policy
    for t in range(NUMBER_OF_RUNS):
        model = PolicyNetwork(num_states, num_actions, num_hidden)
        reinforce = REINFORCE(model,learn_rate)

        # Natural gradient
        optimizer_NG = SGD(model.parameters(), learn_rate)
        # Perform REINFORCE on environment
        episode_returns_policy_gradient_NG, loss_NG = reinforce.run_episodes_policy_gradient(
            env, num_episodes, discount_factor, optimizer_NG)
        NG_results[t] = episode_returns_policy_gradient_NG
        NG_loss[t] = loss_NG


        ## VANILLA GRADIENT
        model_VG = PolicyNetwork(num_states, num_actions, num_hidden)
        reinforce = REINFORCE(model_VG,learn_rate_VG)
        optimizer_VG = torch.optim.SGD(model_VG.parameters(), learn_rate_VG)
        episode_returns_policy_gradient_VG, loss_SGD = reinforce.run_episodes_policy_gradient(
        env, num_episodes, discount_factor, optimizer_VG)
        SGD_results[t] = episode_returns_policy_gradient_VG
        SGD_loss[t] = loss_SGD

    # Calculating average per episode
    average_return_SGD = np.mean(SGD_results, axis=0)
    var_return_SGD = np.std(SGD_results, axis=0)

    average_return_NG = np.mean(NG_results, axis=0)
    var_return_NG = np.std(NG_results, axis=0)

    average_loss_SGD = np.mean(SGD_loss, axis=0)
    var_loss_SGD = np.std(SGD_loss, axis=0)

    average_loss_NG = np.mean(NG_loss, axis=0)  
    var_loss_NG = np.std(NG_loss, axis=0)



    # ## Adam
    # model = PolicyNetwork(num_states, num_actions, num_hidden)
    # reinforce = REINFORCE(model,learn_rate)
    # optimizer_VG = torch.optim.Adam(model.parameters(), learn_rate)
    # episode_returns_policy_gradient_adam = reinforce.run_episodes_policy_gradient(
    #     env, num_episodes, discount_factor, optimizer_VG)

    # Plot loss
    plt.plot(average_return_NG, color="red")
    plt.fill_between(np.arange(num_episodes), average_return_NG - var_return_NG, average_return_NG + var_return_NG, color="red", alpha=0.2)
    plt.plot(average_return_SGD, color="blue")
    plt.fill_between(np.arange(num_episodes), average_return_SGD - var_return_SGD, average_return_SGD + var_return_SGD, color="blue", alpha=0.2)

    plt.title('Average Episode return per episode')
    plt.legend(['Return Natural Policy gradient (SGD)', ' Return Vanilla Policy Gradient (SGD)'])
    plt.show()

    """
    plt.plot(average_loss_SGD, color="red", linestyle='dashed')
    plt.plot(average_loss_NG, color="blue", linestyle='dashed')"""

    


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


run_reinforce(env_name)
