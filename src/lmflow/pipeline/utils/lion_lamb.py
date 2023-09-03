from typing import Tuple, Optional, Callable
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

# functions

def exists(val):
    return val is not None

# update functions

def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2, min_x, max_x):



    # stepweight decay

    p.data.mul_(1 - lr * wd)

    # 不仅保留方向，还保留大小信息
    c = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1)

    u = c.sign()


    # compute the L2 norm of the gradient and the weight
    weight_norm = torch.norm(p.data)
    update_norm = torch.norm(u)


                
    if weight_norm > 0  and update_norm > 0 and not torch.isinf(update_norm):
        trust_ratio = weight_norm / update_norm

    else:
        trust_ratio = 1.0
   

    p.add_(u, alpha = -lr * trust_ratio)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

# class

class Lion_lamb(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: bool = False,
        min_x: float = 0.5,
        max_x: float = 1.5
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        self.min_x = min_x
        self.max_x = max_x
        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

        self.update_fn = update_fn

        if use_triton:
            from lion_pytorch.triton import update_fn as triton_update_fn
            self.update_fn = triton_update_fn

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                # print('count_layer')
                print(p.shape)
                # breakpoint()
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']


                # stepweight decay
                p.data.mul_(1 - lr * wd)
                # Emphasize current step & keep the signed signal 
                c = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1)
                u = c.sign()

                # compute the L2 norm of the gradient and the weight
                weight_norm = torch.norm(p.data)
                update_norm = torch.norm(u)


                if weight_norm > 0  and update_norm > 0 and not torch.isinf(update_norm):
                    trust_ratio = weight_norm / update_norm
                else:
                    trust_ratio = 1.0
            

                p.add_(u, alpha = -lr * trust_ratio)

                # decay the momentum running average coefficient

                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)
                # self.update_fn(
                #     p,
                #     grad,
                #     exp_avg,
                #     lr,
                #     wd,
                #     beta1,
                #     beta2,
                #     min_x = self.min_x,
                #     max_x = self.max_x
                # )

        return loss