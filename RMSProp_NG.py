import torch
from torch.optim import Optimizer

class RMSProp_NG(Optimizer):
    """Implements RMSprop algorithm.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False,ng = False):
        self.f_sum = None
        self.N = 0
        self.ng = ng
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSProp_NG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSProp_NG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        ##### Compute natural gradients
        if self.ng:
            gradients,parameters = [],[]
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    gradients.append(grad.reshape(-1, 1))
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
                natural_grad = torch.mm(fisher_inverse, gradient_vector).squeeze(1)

        # RMSprop
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                if p.grad is None:
                    continue

                if self.ng:
                    p_len = p.data.numel()#np.prod(p.data.shape)
                    grad = natural_grad[i: i + p_len].reshape(p.data.shape)
                    i += p_len
                else:
                    grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(-group['lr'], buf)
                else:
                    p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
