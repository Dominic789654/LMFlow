from typing import Tuple, Optional, Callable
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
import os
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

# functions

def exists(val):
    return val is not None

class Lion_lamb(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        layer_shape:[int] = 0,
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
        self.layer_shape = layer_shape
        defaults = dict(
            lr = lr,
            betas = betas,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        # 获取当前卡的编号
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        
        # 获取分布式训练的总卡数
        world_size = torch.distributed.get_world_size()
        
        
        # 确定每张卡的参数数量
        params_per_card =  len(self.param_groups[0]['params'][0])
        
        # 确定当前卡的参数范围
        start_param = local_rank * params_per_card
        end_param = start_param + params_per_card

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
            
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                p.data.mul_(1 - lr * wd)
                update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1 - beta1).sign_()

                accumulated_param_count = 0
                for layer, shape in self.layer_shape.items():
                    num_params = np.prod(shape)
                    layer_start_global = accumulated_param_count
                    layer_end_global = accumulated_param_count + num_params
                    
                    if layer_start_global < end_param and layer_end_global > start_param:
                        # 计算当前层参数在当前卡上的起始和结束索引
                        start_idx = max(0, layer_start_global - start_param)
                        end_idx = min(params_per_card, layer_end_global - start_param)


                        weight_norm = torch.norm(p.data[start_idx:end_idx])
                        update_norm = torch.norm(update[start_idx:end_idx])

                        if weight_norm > 0 and update_norm > 0 and not torch.isinf(update_norm):
                            trust_ratio = weight_norm / update_norm
                        else:
                            trust_ratio = 1.0
                            # 默认值可以尝试0.1 
                        # breakpoint()
                        grad_norm = torch.norm(p.grad.data[start_idx:end_idx])
                        
                        # for gpt2
                        # if "wte" in layer:
                        #     lr = 1.3*lr
                        # elif "ln_f" in layer:
                        #     lr = 1.3*lr

                        # for llama2
                        if "embed_tokens" in layer:
                            lr = 2*lr
                        elif "weight" in layer:
                            lr = 2*lr
                        p.data[start_idx:end_idx].add_(update[start_idx:end_idx], alpha=-lr * trust_ratio)

                        print_update_norm = torch.norm(update[start_idx:end_idx] * trust_ratio)
                        print(f"\nlocal rank {os.environ.get('LOCAL_RANK', '0')}, layer {layer}, shape {shape}, start {start_idx}, end {end_idx} weight norm {weight_norm}, update norm {print_update_norm}, ratio {trust_ratio}, grad norm {grad_norm}")
                    accumulated_param_count += num_params
                exp_avg.mul_(beta2).add_(grad, alpha = 1 - beta2)

        return loss
