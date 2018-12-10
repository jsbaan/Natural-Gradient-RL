from Gauss import Gauss
import numpy as np
from matplotlib import pyplot as plt
from helper import Helper
from torch import nn
import torch
import torch.nn.functional as F
helper = Helper()

param = 'exp'
episodes = 5000
NUMBER_OF_RUNS = 10
alpha = 0.2

def sigma(sigma):
	if param == 'exp':
		return np.exp(sigma)
	# if param == 'log':
	# 	return np.log(sigma)
	# if param == 'prec':
	# 	return 1/sigma**2
	if param == 'sigma':
		return np.abs(sigma)

def derivative_sigma_param(sigma, mean, action, fisher=False):
	if param == 'exp':
		mean_grad = (action - mean ) / np.exp(sigma)**2
		sigma_grad = -1 + (action - mean)**2 / np.exp(sigma)**2
		# print(sigma)
		if fisher:
			return mean_grad * np.exp(sigma)**2, sigma_grad/2
		return mean_grad, sigma_grad
	# if param = 'log':
	# 	return np.log(sigma)
	# if param = 'prec':
	# 	return 1/sigma**2
	if param == 'sigma':
		mean_grad = (action - mean ) / sigma**2
		sigma_grad = -1/sigma + (action - mean)**2 / sigma**3
		if fisher:
			return mean_grad * sigma**2 , 0.5 * sigma**2 * sigma_grad
		return mean_grad, sigma_grad

# def derivative_gaussian(mean, sigma):
# 	deriv_sigma = -1/(sigma) + 

def rewardfunction(action):
	normal = torch.distributions.Normal(4, 2)
	reward = np.exp(normal.log_prob(action)).numpy()
	return reward


SGD_results = np.zeros((NUMBER_OF_RUNS, episodes))
NG_results = np.zeros((NUMBER_OF_RUNS, episodes))

for t in range(NUMBER_OF_RUNS):
	model = Gauss() 
	returns= []
	losses = []
	for i in range(episodes):
		action, log_action = model.forward(sigma)
		G = rewardfunction(action)
		returns.append(G)
		if model.std < 1e-1:
			model.std = 1e-1
		grad_mean, grad_sigma = derivative_sigma_param(model.std, model.mean, action, fisher=False)

		model.std = model.std + alpha * grad_sigma * G
		model.mean = model.mean + alpha * grad_mean * G

		loss = - G * log_action
		losses.append(loss)
	SGD_results[t] = returns

	returns_NG = []
	model = Gauss() 
	for i in range(episodes):
		action, log_action = model.forward(sigma)
		G = rewardfunction(action)
		returns_NG.append(G)
		if model.std < 1e-1:
			model.std = 1e-1
		grad_mean, grad_sigma = derivative_sigma_param(model.std, model.mean, action, fisher=True)

		model.std = model.std + alpha * grad_sigma * G
		model.mean = model.mean + alpha * grad_mean * G

		loss = - G * log_action
	NG_results[t] = returns_NG

# Calculating average per episode
average_return_SGD = np.mean(SGD_results, axis=0)
var_return_SGD = np.std(SGD_results, axis=0)

average_return_NG = np.mean(NG_results, axis=0)
var_return_NG = np.std(NG_results, axis=0)

plt.plot(average_return_NG, color="red")
plt.fill_between(np.arange(episodes), average_return_NG - var_return_NG, average_return_NG + var_return_NG, color="red", alpha=0.2)

plt.plot(average_return_SGD, color="blue")
plt.fill_between(np.arange(episodes), average_return_SGD - var_return_SGD, average_return_SGD + var_return_SGD, color="blue", alpha=0.2)



plt.title('Average Episode returns and loss per episode')
plt.legend(['Return Natural Policy gradient (SGD)', ' Return Vanilla Policy Gradient (SGD)'])
plt.show()


# plt.plot(helper.smooth(returns, 10))
# plt.show()

# plt.plot(helper.smooth(losses, 10))
# plt.show()



