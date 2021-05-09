import math

import torch
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__ = ('sparseAdaLARS',)


class sparseAdaLARS(Optimizer):
    r"""Implements AdaLARS algorithm.
    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay: weight decay (L2 penalty) (default: 0)
        clamp_value: clamp weight_norm in (0,clamp_value) (default: 10)
            set to a high value to avoid it (e.g 10e3)
    Example:
        >>> import torch_optimizer as optim
        >>> optimizer = optim.AdaLARS(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1904.00962
    Note:
        Reference code: https://github.com/cybertronai/pytorch-lamb
    """

    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
    ) -> None:
        if lr <= 0.0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if eps < 0.0:
            raise ValueError('Invalid epsilon value: {}'.format(eps))
        if weight_decay < 0:
            raise ValueError(
                'Invalid weight_decay value: {}'.format(weight_decay)
            )
        if clamp_value < 0.0:
            raise ValueError('Invalid clamp value: {}'.format(clamp_value))

        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value

        super(sparseAdaLARS, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        r"""Performs a single optimization step.
        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                #when gradients are sparse
                if grad.is_sparse:
                    state = self.state[p]

                    #state initialization
                    if len(state) == 0:
                        state['step'] = 0
                    # Exponential moving average of gradient values
                        state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1

                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values):
                        constructor = grad.new
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor().resize_as_(grad)
                        return constructor(grad_indices, values, size)

                    sum_gradient_sq = state['sum']


                    #Compute denom and accumulate sparse gradients via pytorch API
                    old_sum_values = sum_gradient_sq.sparse_mask(grad)._values()
                    sum_gradient_sq_update_values = grad_values.pow(2)
                    sum_gradient_sq.add_(make_sparse(sum_gradient_sq_update_values))

                    sum_gradient_sq_update_values.add_(old_sum_values)
                    denom = sum_gradient_sq_update_values.sqrt_().add_(group['eps'])
                    del sum_gradient_sq_update_values

                    weight_norm = torch.norm(p.data.sparse_mask(grad)._values())
                    adagrad_step = grad_values/denom
                    adagrad_norm = torch.norm(adagrad_step)

                    if weight_norm == 0 or adagrad_norm == 0:
                        trust_ratio = 1
                    else:
                        trust_ratio = weight_norm / adagrad_norm
                    state['weight_norm'] = weight_norm
                    state['adagrad_norm'] = adagrad_norm
                    state['trust_ratio'] = trust_ratio

                    # Use step size
                    step_size = group['lr'] * trust_ratio

                    p.add_(make_sparse(adagrad_step * step_size))


                #when gradients are dense
                else:
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['sum'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'] += 1

                    # Accumulate the second moment of gradients
                    sum_gradient_sq = state['sum']
                    old_sum_values = sum_gradient_sq
                    sum_gradient_sq_update_values = grad.pow(2)
                    sum_gradient_sq.add_(sum_gradient_sq_update_values)

                    sum_gradient_sq_update_values.add_(old_sum_values)
                    denom = sum_gradient_sq_update_values.sqrt_().add_(group['eps'])
                    del sum_gradient_sq_update_values

                    # Accumulate the second moment of gradients
                    # state['sum'].addcmul_(1, grad, grad)

                    # Compute weight norm
                    weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                    # Use step size
                    step_size = group['lr']

                    # Compute adagrad step
                    adagrad_step = grad / denom
                    if group['weight_decay'] != 0:
                        adagrad_step.add_(p.data, alpha=group['weight_decay'])

                    adagrad_norm = torch.norm(adagrad_step)
                    if weight_norm == 0 or adagrad_norm == 0:
                        trust_ratio = 1
                    else:
                        trust_ratio = weight_norm / adagrad_norm
                    state['weight_norm'] = weight_norm
                    state['adagrad_norm'] = adagrad_norm
                    state['trust_ratio'] = trust_ratio

                    p.data.add_(adagrad_step, alpha=-step_size * trust_ratio)

        return loss