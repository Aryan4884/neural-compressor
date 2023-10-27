# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class BaseConfig():
  def __init__(self):
      self.scope = {}

  def __add__(self, obj):
      self.scope = {**self.scope, **obj.scope}

class FP32Config(BaseConfig):
  def __init__(self, scope):
      super().__init__()
      self.scope = {'fp32': scope}

class FP8QuantConfig(BaseConfig):
  def __init__(self, scheme, scope):
      super().__init__()
      assert(scheme in ['fp8_e4m3', 'fp8_e3m4', 'fp8_e5m2']), "The FP8 configuration is wrong! Please double check."
      self.scope = {scheme: scope}


from neural_compressor.common.constant import MAX_TRIALS_EACH_SAMPLER


# Place configs that can be applied to all adaptors here.
# These configs are designed for advanced user to customized tuning process with more flexibility.
class BasicSamplerConfig:
    priority = 100
    max_trials = MAX_TRIALS_EACH_SAMPLER


class SmoothQuantSamplerConfig(BasicSamplerConfig):
    alpha = [0.5]


class OpTypeWiseSamplerConfig(BasicSamplerConfig):
    priority = 10
    op_types = []


class OptimizationLevelSamplerConfig(BasicSamplerConfig):
    priority = 100
    optimization_levels = []


basic_sampler_config = BasicSamplerConfig()
smooth_quant_sampler_config = SmoothQuantSamplerConfig()
op_type_wise_sampler_config = OpTypeWiseSamplerConfig()
optimization_level_sampler_config = OptimizationLevelSamplerConfig()


class TuningConfig:
    """Tuning Config class.

    #TODO(Yi) should we use TuningConfig to replace TuningCriterion and AccuracyCriterion?
    TuningConfig class is used to configure the trials order, accuracy constraint and exit policy.
    Note: The TuningConfig class merges the main functionalities of TuningCriterion and AccuracyCriterion of INC 2.x.

    Attributes for trials order:
        - quant_level
        - sampler[New introduced to replace `strategy`]
    Attributes for accuracy criterion:
        - higher_is_better
        - relative
        - tolerable_loss
    Attributes for exit policy:
        - objective
        - timeout
        - max_trials
    """

    pass


class TuningCriterion:
    """Class for Tuning Criterion.

    Args:
        strategy: Strategy name used in tuning. Please refer to docs/source/tuning_strategies.md.
        strategy_kwargs: Parameters for strategy. Please refer to docs/source/tuning_strategies.md.
        objective: String or dict. Objective with accuracy constraint guaranteed. String value supports
                  'performance', 'modelsize', 'footprint'. Default value is 'performance'.
                   Please refer to docs/source/objective.md.
        timeout: Tuning timeout (seconds). Default value is 0 which means early stop.
        max_trials: Max tune times. Default value is 100. Combine with timeout field to decide when to exit.

    Example::
        from neural_compressor.config import TuningCriterion

        tuning_criterion=TuningCriterion(
            timeout=0,
            max_trials=100,
            strategy="basic",
            strategy_kwargs=None,
        )
    """

    def __init__(self, strategy="basic", strategy_kwargs=None, timeout=0, max_trials=100, objective="performance"):
        """Init a TuningCriterion object."""
        self.strategy = strategy
        self.timeout = timeout
        self.max_trials = max_trials
        self.objective = objective
        self.strategy_kwargs = strategy_kwargs


tuning_criterion = TuningCriterion()


class AccuracyCriterion:
    """Class of Accuracy Criterion.

    Args:
        higher_is_better(bool, optional): This flag indicates whether the metric higher is the better.
                                          Default value is True.
        criterion:(str, optional): This flag indicates whether the metric loss is 'relative' or 'absolute'.
                                   Default value is 'relative'.
        tolerable_loss(float, optional): This float indicates how much metric loss we can accept.
                                         Default value is 0.01.

    Example::

        from neural_compressor.config import AccuracyCriterion

        accuracy_criterion = AccuracyCriterion(
            higher_is_better=True,  # optional.
            criterion='relative',  # optional. Available values are 'relative' and 'absolute'.
            tolerable_loss=0.01,  # optional.
        )
    """

    def __init__(self, higher_is_better=True, criterion="relative", tolerable_loss=0.01):
        """Init an AccuracyCriterion object."""
        self.higher_is_better = higher_is_better
        self.criterion = criterion
        self.tolerable_loss = tolerable_loss


accuracy_criterion = AccuracyCriterion()

# TODO(Yi) The above code will be removed


########################################################
##############    Algo-related configs    ##############
########################################################
"""
       user_model --.
                     \
user_sq_config -- [adaptor] -> per_op_sq_config  -> [tuner] -> tuner expand it and pass each combination to [adaptor] 
                                                       ^                         |
                                                       |                         |
                                                       |           adaptor apply the specific config
                                                       |                         |
                                                       ------------    return the eval result to tuner
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Type, Union


class Priority(Enum):
    """The priority of algorithm.

    The algorithms with higher priority(value) will be tried  first.
    Users can also assign an integer or floating point number.
    """

    LOW = 1
    MEDIUM = 1000
    HIGH = 1000_000


@dataclass
class Parameter:
    """Set the tunable parameters of a given compression algorithm.

    Args:
        name: a unique string to present the tunable parameter
        values: the possible values of this parameter

    Examples:
        >>> sq_folding = Parameter('folding', [True, False])
        >>> sq_alpha = Parameter('alpha', 0.5)
    """

    name: str
    values: Union[Any, List[Any]]


class AlgorithmConfig(ABC):
    """The base class for Algorithm.
    
    Example usage::
        #############################
        #####  End User Usage #######
        #############################

        from neural_compressor.torch.quantization import get_default_sq_config, quantize

        sq_config = get_default_sq_config(backend="ipex")
        ##############################################################################
        # Demonstrate how does the user modify the config
        
        # User can increase priority to try SQ tuning first.
        sq_config.priority = 1_000_000

        # User can customize the alpha list.
        from neural_compressor.common import Parameter
        sq_config.alpha = Parameter('alpha', [0.1, 0.9])

        # User can also assign a new expand func
        def new_expand_func():
            return None
        sq_config.expand_config = new_expand_func

        # User can add filter functions for expand func.
        def new_filter_func(config):
            if config.folding and config.alpha < 0.1:
                return False

        sq_config.add_constrains(new_filter_func)

        # The above code is only for the user wants to modify the config
        ##############################################################################

        # quantize model
        model_fp = UserModel()
        q_model = quantize(model_fp, config=sq_config)
    """

    def __init__(
        self,
        params_lst: Union[Parameter, List[Parameter], None] = None,
        white_lst: Union[str, List[str], None] = None,
        black_lst: Union[str, List[str], None] = None,
        priority: Union[int, float, None] = 0,
        constraint_func_lst: Union[List[Callable], None] = None,
    ):
        """The base class for algorithm configuration.

        Args:
            params_lst: the tunable parameter list. Defaults to None.
            white_lst: the list of operator that this algorithm can be applied. Defaults to None.
            black_lst: the list of operator that this algorithm can't be applied. Defaults to None.
            priority: the priority of this algorithm when tuning. Defaults to 0.
            constraint_func_lst: A list of constraint functions used to filter some invalid combinations when expanding.
                Defaults to None.
        """
        self.params_lst = params_lst
        self.white_lst = white_lst
        self.black_lst = black_lst
        self.priority = priority
        self.constraint_func_lst = constraint_func_lst
        self._parameters = dict()
        self._post_init()

    def _post_init(self):
        if self.params_lst is not None:
            for param in self.params_lst:
                self.register_parameter(param.name, param.values)

    def __setattr__(self, name, value):
        """
        Examples:
            >>> sq_algo = AlgorithmConfig()
            >>> sq_algo.alpha = Parameter('alpha', [0.2, 0.3])
        """
        if isinstance(value, Parameter):
            self.register_parameter(name, value)

    def register_parameter(self, name, value) -> None:
        """Register tunable parameters.

        Args:
            name: parameter name
            value: parameter values.
        """
        self._parameters[name] = value

    @abstractmethod
    def expand_config(self, model):
        """Generates a set of valid configurations to specify the behavior of **each operation**.

        The content of each configuration should includes:
            - The exact value of each tunable parameter.
            - Additional properties required when applying the algorithm

        # TODO(Yi) Need model to expand the config?
        The adaptor needs implement it.
        The different algo config may have different `expand` method.
        This function expand all valid combinations for tunable params.
        This function was called by tuner to search the best config.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(
        self, user_config: Type["AlgorithmConfig"], model=None
    ) -> Type["AlgorithmConfig"]:
        """Merge the options of tunable params according to user config, kernel config and user model.

        The different algo config may have different `merge` method.
          user config  -.
           user model   ----> tuning space for this algorithm
        kernel config  -'
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_default_config(cls):
        """Construct the kernel/algorithm capability and scope.

        The Function includes:
            - tunable parameters, like quantization scheme, optimization level, and so on.
            - The scope to which operators can be applied.

        Note: The functionality is similar to the framework YAML in INC 2.0.

        Raises:
            NotImplementedError: framework should implement it.
        """
        raise NotImplementedError

    def add_constraints(
        self, constraint_func: Union[Callable, List[Callable], None] = None
    ):
        """Add constraint functions to filter some invalid combinations.

        Args:
            constraint_func: constraint function. Defaults to None.
        """
        if constraint_func is not None:
            if isinstance(constraint_func, list):
                self.constraint_func_lst.extend(constraint_func)
            else:
                self.constraint_func_lst.append(constraint_func)


########################################################
##############    Tuner pseudo code        ##############
########################################################
# TODO(Yi) remove it before merge.
# class Tuner:
#     def __init__(self) -> None:
#         self.algo_config_lst = []
#         self.adaptor = None

#     def search(self):
#         for algo_config in self.algo_config_lst:
#             for config in algo_config.expand_config():
#                 q_model = self.adaptor.apply(config)
#                 eval_res = self.evaluate(q_model)
#                 if self.need_stop(eval_res):
#                     return q_model
