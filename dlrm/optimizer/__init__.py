"""torch-optimizer -- collection of of optimization algorithms for PyTorch.

API and usage patterns are the same as `torch.optim`__

Example
-------

>>> import torch_optimizer as optim
# model = ...
>>> optimizer = optim.DiffGrad(model.parameters(), lr=0.001)
>>> optimizer.step()

See documentation for full list of supported optimizers.

__ https://pytorch.org/docs/stable/optim.html#module-torch.optim
"""
'''from typing import Dict, List, Type

from pytorch_ranger import Ranger, RangerQH, RangerVA
from torch.optim.optimizer import Optimizer


from .adalars import AdaLARS


__all__ = (
    'AdaLARS',
    # utils
    'get',
)
__version__ = '0.1.0'


_package_opts = [
    AdaLARS,
]  # type: List[Type[Optimizer]]


_NAME_OPTIM_MAP = {
    opt.__name__.lower(): opt for opt in _package_opts
}  # type: Dict[str, Type[Optimizer]]


def get(name: str) -> Type[Optimizer]:
    r"""Returns an optimizer class from its name. Case insensitive.

    Args:
        name: the optimizer name.
    """
    optimizer_class = _NAME_OPTIM_MAP.get(name.lower())
    if optimizer_class is None:
        raise ValueError('Optimizer {} not found'.format(name))
    return optimizer_class'''
