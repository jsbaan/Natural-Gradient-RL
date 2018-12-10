from Gauss import Gauss
import numpy as np
from matplotlib import pyplot as plt
from helper import Helper
from torch import nn
import torch
import torch.nn.functional as F
helper = Helper()

param = 'exp'
episodes = 1000
NUMBER_OF_RUNS = 20
alpha_van = 0.5
alpha_nat = alpha_van
smooth = 10

# alpha voor exp was 0.2 voor allebei

def sigma(sigma):
	if param == 'exp':
		return np.exp(sigma)
	if param == 'prec':
		return 1/sigma**2
	if param == 'sigma':
		return np.abs(sigma)

def derivative_sigma_param(sigma, mean, action, fisher_mean_sum, fisher_sigma_sum, fisher_N, fisher=False):
	if param == 'exp':
		mean_grad = (action - mean ) / np.exp(sigma)**2
		sigma_grad = -1 + (action - mean)**2 / np.exp(sigma)**2
		if fisher:
			fisher_N += 1
			fisher_mean_sum += 1/np.exp(sigma)**2 # 1/sigma(theta)
			fisher_mean_mean = fisher_mean_sum / fisher_N
			fisher_mean_inverse = 1/fisher_mean_mean
			
			fisher_sigma_sum = fisher_sigma_sum
			fisher_sigma_inverse = 1/2
			return mean_grad * fisher_mean_inverse, sigma_grad * fisher_sigma_inverse,  fisher_mean_sum, fisher_sigma_sum, fisher_N
		return mean_grad, sigma_grad
	if param == 'prec':
		mean_grad = (action - mean ) / (1/sigma**2)**2
		sigma_grad = 2/sigma - sigma**3 * (action - mean)**2
		if fisher:
			return mean_grad * (1/sigma**2)**2, sigma_grad * sigma**2/3
		return mean_grad, sigma_grad
	if param == 'sigma':
		mean_grad = (action - mean ) / sigma**2
		sigma_grad = -1/sigma + (action - mean)**2 / sigma**3
		if fisher:
			return mean_grad * sigma**2 , 0.5 * sigma**2 * sigma_grad
		return mean_grad, sigma_grad


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
	fisher_mean_sum = 0
	fisher_sigma_sum = 0
	fisher_N = 0 
	for i in range(episodes):
		action, log_action = model.forward(sigma)
		G = rewardfunction(action)
		returns.append(G)
		if model.std < 1e-1:
			model.std = 1e-1
		grad_mean, grad_sigma = derivative_sigma_param(model.std, model.mean, action, fisher_mean_sum, fisher_sigma_sum, fisher_N, fisher=False)

		model.std = model.std + alpha_van * grad_sigma * G
		model.mean = model.mean + alpha_van * grad_mean * G

		loss = - G * log_action
		losses.append(loss)
	# print(model.mean, model.std)
	SGD_results[t] = returns

	returns_NG = []
	model = Gauss()
	fisher_mean_sum = 0
	fisher_sigma_sum = 0
	fisher_N = 0 
	for i in range(episodes):
		action, log_action = model.forward(sigma)
		G = rewardfunction(action)
		returns_NG.append(G)
		if model.std < 1e-1:
			model.std = 1e-1
		grad_mean, grad_sigma, fisher_mean_sum, fisher_sigma_sum, fisher_N = derivative_sigma_param(model.std, model.mean, action, fisher_mean_sum, fisher_sigma_sum, fisher_N, fisher=True)


		model.std = model.std + alpha_nat * grad_sigma * G
		model.mean = model.mean + alpha_nat * grad_mean * G

		loss = - G * log_action

	NG_results[t] = returns_NG

# Calculating average per episode
average_return_SGD = helper.smooth(np.mean(SGD_results, axis=0), smooth)
var_return_SGD = helper.smooth(np.std(SGD_results, axis=0), smooth)

average_return_NG = helper.smooth(np.mean(NG_results, axis=0), smooth)
var_return_NG = helper.smooth(np.std(NG_results, axis=0), smooth)

print(average_return_SGD[len(average_return_SGD)-1])
print(average_return_NG[len(average_return_NG)-1])

plt.plot(average_return_NG, color="red")
plt.fill_between(np.arange(len(average_return_NG)), average_return_NG - var_return_NG, average_return_NG + var_return_NG, color="red", alpha=0.2)

plt.plot(average_return_SGD, color="blue")
plt.fill_between(np.arange(len(average_return_SGD)), average_return_SGD - var_return_SGD, average_return_SGD + var_return_SGD, color="blue", alpha=0.2)



plt.title('Average Episode returns and loss per episode')
plt.legend(['Return Natural Policy gradient (SGD)', ' Return Vanilla Policy Gradient (SGD)'], loc = 4)
plt.show()


# plt.plot(helper.smooth(returns, 10))
# plt.show()

# plt.plot(helper.smooth(losses, 10))
# plt.show()



