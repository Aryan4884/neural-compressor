# TODO(Yi) This file maybe moved to neural_compressor.torch

from typing import List, Union

from torch.nn import Module

from neural_compressor.common import AlgorithmConfig, Tuner
from neural_compressor.common.utility import to_list


def _merge_user_configs_with_default_configs(
    user_configs: Union[AlgorithmConfig, List[AlgorithmConfig]]
):
    user_configs = to_list(user_configs)
    merged_configs = []
    for user_config in user_configs:
        default_config = user_config.get_default_config()
        merged_configs.append(default_config.merge(user_config))
    return merged_configs


class Quantizer:
    def __init__(self, model, config) -> None:
        self.model = model
        self.config = config

    def quantize(self):
        """Quantize the model."""
        return None

    def _need_tune(self):
        return True

    def _tuning(self):
        merged_configs = _merge_user_configs_with_default_configs(self.config)
        if self._need_tune():
            tuner = Tuner(self, merged_configs)
            return tuner.search()


def quantize(model: Module, config: Union[AlgorithmConfig, List[AlgorithmConfig]]):
    """The main entry to quantize model."""
    # TODO(Yi) how to pass the objectives and stop-related configs, such as the number of trials, time limitation and so on.
    quantizer = Quantizer(model, config)
    return quantizer.quantize()
