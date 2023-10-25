import math
import torch
from torch.optim.optimizer import Optimizer
import numpy as np
import os 

class AdamW(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, layer_shape=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        self.layer_shape = layer_shape if layer_shape is not None else {}

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

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
            for p in group['params']:

                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]


                # print gradient norm
                # 获取当前卡的编号
                local_rank = int(os.environ.get('LOCAL_RANK', '0'))
                
                # 获取分布式训练的总卡数
                world_size = torch.distributed.get_world_size()
                
                
                # 确定每张卡的参数数量
                params_per_card =  len(self.param_groups[0]['params'][0])
                
             

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom) )

                # 确定当前卡的参数范围
                start_param = local_rank * params_per_card
                end_param = start_param + params_per_card
                accumulated_param_count = 0
                for layer, shape in self.layer_shape.items():
                    num_params = np.prod(shape)
                    layer_start_global = accumulated_param_count
                    layer_end_global = accumulated_param_count + num_params
                    
                    if layer_start_global < end_param and layer_end_global > start_param:
                        # 计算当前层参数在当前卡上的起始和结束索引
                        start_idx = max(0, layer_start_global - start_param)
                        end_idx = min(params_per_card, layer_end_global - start_param)

                        grad_norm = torch.norm(grad[start_idx:end_idx])
                        weight_norm = torch.norm(p.data[start_idx:end_idx])
                        print(f"\nlocal rank {os.environ.get('LOCAL_RANK', '0')}, layer {layer}, shape {shape}, start {start_idx}, end {end_idx}, grad norm {grad_norm}, weight norm {weight_norm}, update norm {torch.norm(exp_avg[start_idx:end_idx] /denom[start_idx:end_idx])}")

                    accumulated_param_count += num_params
        return loss