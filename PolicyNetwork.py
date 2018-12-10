import torch
from torch import nn
import torch.nn.functional as F

# seed = 20
# torch.manual_seed(seed)

class PolicyNetwork(nn.Module):

    def __init__(self, num_states, num_actions, num_hidden=128):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(num_states, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_actions)

    def forward(self, x):
        i2h = F.relu(self.l1(x))
        log_softmax = nn.LogSoftmax(1)
        output = self.l2(i2h)
        output = log_softmax(output)
        return output
