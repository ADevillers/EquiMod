import torch
from torch.optim.optimizer import Optimizer, required



class LARSW(Optimizer):
    """
    A LARS PyTorch implementation.
    W stand for weighted, as the `lars_weight` parameter allows to switch from 0 -> Normal SGD;  1 -> Normal LARS
    """
    def __init__(self, params, lr=required, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=False, eta=0.001, lars_weight=1.0):
        if lr is not required and lr < 0.0:
            raise ValueError('Invalid learning rate: {} - should be >= 0.0'.format(lr))
        if momentum < 0.0:
            raise ValueError('Invalid momentum: {} - should be >= 0.0'.format(momentum))
        if weight_decay < 0.0:
            raise ValueError('Invalid weight decay: {} - should be >= 0.0'.format(weight_decay))
        if eta < 0.0 or eta >= 1.0:
            raise ValueError('Invalid eta: {} - should be in [0.0, 1.0['.format(eta))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
            eta=eta,
            lars_weight=lars_weight
        )

        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']
            eta = group['eta']
            lars_weight = group['lars_weight']

            for p in group['params']:
                if p.grad is None: continue
                param = p.data
                grad = p.grad.data
                state = self.state[p]

                grad.add_(param, alpha=weight_decay)

                local_lr = 1.0
                w_norm = param.norm()
                g_norm = grad.norm()
                if w_norm * g_norm > 0:
                    local_lr = eta*w_norm/g_norm
                scaled_lr = lr*(1 - lars_weight) + lr*local_lr*lars_weight

                local_grad = scaled_lr*grad
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(local_grad).detach()
                else:
                    buf = state['momentum_buffer'].mul_(momentum).add_(local_grad, alpha=1 - dampening)

                if nesterov:
                    final_grad = local_grad.add(buf, alpha=momentum)
                else:
                    final_grad = buf

                p.data.add_(-final_grad)

        return loss
