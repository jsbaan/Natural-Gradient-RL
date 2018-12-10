import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
class Gauss():
	def __init__(self):
		self.mean = np.random.uniform(-1,1)
		self.std = np.random.uniform(0,2)

	def forward(self, function):
		normal = torch.distributions.Normal(self.mean, function(self.std))
		action = normal.sample().numpy()
		log_prob = normal.log_prob(action).numpy()
		return action, log_prob
