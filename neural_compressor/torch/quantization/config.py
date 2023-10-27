from typing import Callable, List, Optional, Type, Union

import torch
from torch.nn import Module

from neural_compressor.common.config import AlgorithmConfig, Parameter, Priority

STR_OR_MODULE = Union[str, Module]


class SmoothQuantConfig(AlgorithmConfig):
    def __init__(
        self,
        params_lst: List[Parameter] = None,
        white_lst: Union[STR_OR_MODULE, List[STR_OR_MODULE]] = None,
        black_lst: Union[STR_OR_MODULE, List[STR_OR_MODULE]] = None,
        priority: Union[int, float] = Priority.HIGH,
        constraint_func_lst: Union[List[Callable], None] = None,
    ):
        """The smooth quant algorithm configuration class.

        Args:
            params_lst: the tunable parameter list of smooth quant algorithm. Defaults to None.
            white_lst: the list of operator that the smooth quant can be applied. Defaults to None.
            black_lst: the list of operator that the smooth quant  can't be applied. Defaults to None.
            priority: the priority of the smooth quant when tuning. Defaults to HIGH.
            constraint_func_lst: A list of constraint functions used to filter some invalid combinations when expanding.
                Defaults to None.
        """
        super().__init__(
            params_lst, white_lst, black_lst, priority, constraint_func_lst
        )

    def expand_config(self, model):
        # TODO(Xin) implement it
        pass

    def merge(
        self, user_config: Type["SmoothQuantConfig"]
    ) -> Type["SmoothQuantConfig"]:
        # TODO(Xin) implement it
        pass

    @classmethod
    def get_default_config(cls, backend=None):
        """Construct the tunable parameters and scope for smooth quant.

        Tunable parameters:
            - folding: True or False
            - alpha: a float point or a list of float point
            - white_lst: Linear module and Conv2d module

        Args:
            backend: `default` or `ipex`. Defaults to None.

        Returns:
            The instance of SmoothQuantConfig.
        """
        if backend is None:
            import numpy as np

            sq_alpha = Parameter("alpha", np.arange(0.1, 0.5, 0.1).tolist())
            sq_folding = Parameter("folding", [True, False])
            default_sq_config = SmoothQuantConfig(
                [sq_alpha, sq_folding], white_lst=[torch.nn.Linear, torch.nn.Conv2d]
            )
            return default_sq_config


def get_default_sq_config(backend: Optional[str] = None):
    """Get the default smooth quant config.

    Args:
        backend: the backend. Defaults to None.

    Returns:
        Return the default smooth quant config
    """
    return SmoothQuantConfig.get_default_config(backend)
