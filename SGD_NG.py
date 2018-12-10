import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np
import sys
class SGD_NG(Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        self.f_sum = None
        self.N = 0
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_NG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_NG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            # Convert gradients and parameters to single vector
            gradients = []
            parameters = []
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                # Perform momentum or nesterov on gradients
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                gradients.append(d_p.reshape(-1, 1))
                parameters.append(p.data.reshape(-1, 1))

            parameter_vector = torch.cat(parameters, 0)
            gradient_vector = torch.cat(gradients, 0)

            # Add robustifying constant
            robust_c = torch.eye(gradient_vector.shape[0]) * 1e-3
            fisher_sample = torch.mm(gradient_vector, gradient_vector.t())
            if self.N == 0:
                self.f_sum = fisher_sample
            else:
                self.f_sum += fisher_sample
            self.N += 1

            # use f_sum/N as consistent estimate for F
            fisher_inverse = (self.f_sum/self.N + robust_c).inverse()
            parameter_vector = parameter_vector.squeeze(1)
            scaled_grad = torch.mm(fisher_inverse, gradient_vector).squeeze(1)

            updated_weight = parameter_vector - group['lr'] * scaled_grad
            i = 0
            for p in group['params']:
                p_len = np.prod(p.data.shape)
                # update weights
                p.data = updated_weight[i: i + p_len].reshape(p.data.shape)
                i += p_len
        return loss
