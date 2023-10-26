from typing import Callable, List, Type, Union

from torch.nn import Module

from neural_compressor.common.config import AlgorithmConfig, Parameter

STR_OR_MODULE = Union[str, Module]


class SmoothQuantConfig(AlgorithmConfig):
    def __init__(
        self,
        params_lst: List[Parameter] = None,
        white_lst: Union[STR_OR_MODULE, List[STR_OR_MODULE]] = None,
        black_lst: Union[STR_OR_MODULE, List[STR_OR_MODULE]] = None,
        priority: Union[int, float] = 0,
    ):
        super().__init__(params_lst, white_lst, black_lst, priority)
        # TODO(framework developer) to customize the config
        pass

    def expand_config(self, model):
        return super().expand_config(model)

    def merge(
        self, user_config: Type["SmoothQuantConfig"]
    ) -> Type["SmoothQuantConfig"]:
        pass

    def add_constrains(self, constrain_func: Union[Callable, List[Callable]] = None):
        pass

    @classmethod
    def get_default_config(cls, backend=None):
        if backend is None:
            import numpy as np

            sq_alpha = Parameter("alpha", np.arange(0.1, 0.5, 0.1).tolist())
            sq_folding = Parameter("folding", [True, False])
            default_sq_config = SmoothQuantConfig(
                [sq_alpha, sq_folding], white_lst=["Linear", "Conv2d"]
            )
            return default_sq_config


def get_default_sq_config(backend: str = None):
    """Get the default smooth quant config.

    Args:
        backend: the backend. Defaults to None.

    Returns:
        Return the default smooth quant config
    """
    return SmoothQuantConfig.get_default_config(backend)
