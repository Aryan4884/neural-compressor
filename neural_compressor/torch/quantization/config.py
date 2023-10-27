from enum import Enum, auto
from typing import Union

import torch
from torch.nn import Module

from neural_compressor.common.config import (
    AlgorithmCap,
    AlgorithmConfig,
    Options,
    Priority,
    register_algorithm,
    register_algorithm_config,
)


class DataType(Enum):
    FP8 = auto()
    INT8 = auto()
    FP32 = auto()
    BF16 = auto()


STR_OR_MODULE = Union[str, Module]


@register_algorithm_config(algo_name="smooth_quant")
@dataclass
class SmoothQuantConfig(AlgorithmConfig):
    def __init__(
        self,
        dtype=[DataType.FP8, DataType.INT8],
        alpha=[0.5],
        folding=[True, False],
        white_lst=[torch.nn.Linear, torch.nn.Conv2d],
        black_lst=None,
    ):
        super().__init__(white_lst, black_lst)
        self.dtype = dtype
        self.alpha = alpha
        self.folding = folding


@register_algorithm(algo_name="smooth_quant")
class SmoothQuantCap(AlgorithmCap):
    """Register the tunable parameters and scope for smooth quant."""

    def __init__(self) -> None:
        self.dtype: Options = Options(
            default_options=[DataType.FP8, DataType.INT8],
            check_func_lst=lambda x: x in [DataType.FP8, DataType.INT8],
        )
        self.alpha: Options = Options(
            default_options=[0.5], check_func_lst=[lambda x: 0 < x and x < 1]
        )
        self.alpha: Options = Options(
            default_options=[True, False], check_func_lst=lambda x: isinstance(x, bool)
        )
        self.white_lst: Options = Options(
            default_options=[torch.nn.Linear, torch.nn.Conv2d],
            check_func_lst=lambda x: isinstance(x, torch.nn.Module),
        )
        self.priority = Priority.HIGH


def get_default_sq_config(backend="default"):
    """Return the default smooth quant config to user.

    Args:
        backend: the backend, 'default' or 'ipex'. Defaults to 'default'.

    Returns:
        return the default smooth quant config according to the backend.
    """
    if backend == "default":
        default_sq_config = SmoothQuantConfig()
        return default_sq_config


#############################
#####  End User Usage #######
#############################

from neural_compressor.torch.quantization import get_default_sq_config, quantize

sq_config = get_default_sq_config(backend="ipex")

##############################################################################
# Demonstrate how does the user modify the config
# User can customize the alpha list
sq_config.alpha = [0.1, 0.9]

# case 2
# from neural_compressor.torch.quantization import SmoothQuantConfig
# sq_config = SmoothQuantConfig(alpha=[0.1, 0.9])

# The above code is only for the user wants to modify the config
##############################################################################

# quantize model
model_fp = UserModel()
q_model = quantize(model_fp, config=sq_config)
