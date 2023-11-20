import torch
from torch.optim.optimizer import Optimizer

class Adagrad(Optimizer):
    def __init__(self, params, lr=1e-2, eps=1e-10,layer_shape=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, eps=eps)
        super(Adagrad, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_squares = state['sum']
                lr = group['lr']
                eps = group['eps']

                sum_squares.addcmul_(1, grad, grad)
                std = sum_squares.sqrt().add_(eps)
                p.data.addcdiv_(-lr, grad, std)

        return loss
