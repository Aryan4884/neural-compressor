# TODO(Yi) This file maybe moved to neural_compressor.torch

from typing import List, Union

from torch.nn import Module

from neural_compressor.common import AlgorithmConfig, Tuner


def _merge_user_configs_with_default_configs(
    user_configs: Union[AlgorithmConfig, List[AlgorithmConfig]]
):
    framework_cap_lst = get_all_cap_of_fwk()
    merged_configs = []
    for framework_cap in framework_cap_lst:
        merged_configs.append(framework_cap.merge(user_configs))
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
