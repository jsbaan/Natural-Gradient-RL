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
helper = Helper()
# Create environment
env_name = "CartPole-v0"


def run_reinforce(env_name):
    # Define parameters
    env = gym.envs.make(env_name)
    num_actions = env.action_space.n
    num_states = env.reset().shape[0]
    num_runs = 20
    num_episodes = 300
    num_hidden = 64
    discount_factor = 0.99
    print("Number of states: ", num_states)
    print("Number of actions: ", num_actions)

    # optim_list = ["SGD_NG", "SGD", "ADAM"]
    # optim_list = ["SGD_NG", "SGD_NG_1"]
    optim_list = ["RMS_.01","RMS_.005","RMS_.001"]
    # optim_list = ["RMSNG_.01","RMSNG_.001","RMSNG_.0001"]


    episode_return_list = {}

    for optim in optim_list:
        print('--- %s ---' % optim)
        env = gym.envs.make(env_name)
        random.seed(10)
        episode_return_list[optim] = np.zeros((1, num_episodes))

        # Rerun num_runs times in order to validate
        for i in range(num_runs):
            print('Run: {}'.format(i))
            env.seed(random.randint(0, 100))

            # Initialize policy
            model = PolicyNetwork(num_states, num_actions, num_hidden)

            # Define optimizer:
            if 'RMS' in optim:
                learn_rate = float(optim.split('_')[1])
                if 'NG' in optim:
                    optimizer = RMSProp_NG(model.parameters(), learn_rate, ng=True)
                else:
                    optimizer = RMSProp_NG(model.parameters(), learn_rate, ng=False)

            elif optim == "SGD_NG":
                learn_rate = 0.01
                optimizer = SGD_NG(model.parameters(), learn_rate)
            elif optim == "SGD":
                learn_rate = 0.0001
                optimizer = torch.optim.SGD(model.parameters(), learn_rate)
            elif optim == "ADAM":
                learn_rate = 0.01
                optimizer = torch.optim.Adam(model.parameters(), learn_rate)
            else:
                raise ValueError('Not a known optimizer')

            reinforce = REINFORCE(model, optimizer)
            result = np.array(reinforce.run_episodes_policy_gradient(
                env, num_episodes, discount_factor))

            # stack observations
            episode_return_list[optim] = np.vstack((episode_return_list[optim], result))

        # delete first row with zeros:
        episode_return_list[optim] = episode_return_list[optim][1:, :]

    # Plot loss
    smooth_ep = 10
    for optim_name, episode_returns in episode_return_list.items():
        mean = np.mean(episode_returns, 0)
        median = np.median(episode_returns, 0)
        sd = np.sqrt(np.var(episode_returns, 0))
        # fill (x-range, lower, upper)
        plt.fill_between(helper.smooth(range(len(mean)), smooth_ep),
                         helper.smooth(mean - sd, smooth_ep), helper.smooth(mean + sd, smooth_ep),
                         alpha=0.5)
        plt.plot(helper.smooth(range(len(median)), smooth_ep), helper.smooth(median, smooth_ep),'-', label=str(optim_name + ' median'))
        # plt.plot(helper.smooth(range(len(mean)), smooth_ep), helper.smooth(mean, smooth_ep), label=str(optim_name + ' (SD)'))
        plt.title('Episode returns per episode')
        plt.legend()
    plt.show()


# run_actorcritic(env_name)
run_reinforce(env_name)
