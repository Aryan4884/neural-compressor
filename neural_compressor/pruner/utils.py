"""prune utils."""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import yaml

try:
    from neural_compressor.conf.dotdict import DotDict
except:
    from .dot_dict import DotDict  ##TODO
from .logger import logger


class WeightPruningConfig:
    """
    similiar to torch optimizer's interface
    """

    def __init__(self, pruning_configs=[{}],  ##empty dict will use global values
                 target_sparsity=0.9, pruning_type="snip_momentum", pattern="4x1", op_names=[],
                 excluded_op_names=[],
                 start_step=0, end_step=0, pruning_scope="global", pruning_frequency=1,
                 min_sparsity_ratio_per_op=0.0, max_sparsity_ratio_per_op=0.98,
                 sparsity_decay_type="exp", pruning_op_types=['Conv', 'Linear'],
                 **kwargs):
        self.pruning_configs = pruning_configs
        self._weight_compression = DotDict({
            'target_sparsity': target_sparsity,
            'pruning_type': pruning_type,
            'pattern': pattern,
            'op_names': op_names,
            'excluded_op_names': excluded_op_names,  ##global only
            'start_step': start_step,
            'end_step': end_step,
            'pruning_scope': pruning_scope,
            'pruning_frequency': pruning_frequency,
            'min_sparsity_ratio_per_op': min_sparsity_ratio_per_op,
            'max_sparsity_ratio_per_op': max_sparsity_ratio_per_op,
            'sparsity_decay_type': sparsity_decay_type,
            'pruning_op_types': pruning_op_types,
            ##reg_type=None, reduce_type="mean", parameters={"reg_coeff": 0.0}
            ##'resume_from_pruned_checkpoint': resume_from_pruned_checkpoint  ##resume_from_pruned_checkpoint
        })
        self._weight_compression.update(kwargs)

    @property
    def weight_compression(self):
        return self._weight_compression

    @weight_compression.setter
    def weight_compression(self, weight_compression):
        self._weight_compression = weight_compression


def check_config(prune_config):
    """Functions that check key-value is valid to run Pruning object.
    
    Args:
        prune_config: A config dict object. Contains Pruning parameters and configurations.
        
    Returns:
        None if everything is correct.
        
    Raises:
        AssertionError.
    """
    assert prune_config['start_step'] >= 0, "start_step should be greater than 0"
    assert prune_config['end_step'] >= -1, "end_step should be greater than 0"
    assert prune_config['end_step'] >= prune_config['start_step'], \
        "end_step should be greater than start_step"
    assert prune_config['target_sparsity'] >= 0 and prune_config['target_sparsity'] < 1.0, \
        "begin_pruning_step should be in range [0,1)"
    assert prune_config['pruning_frequency'] > 0, "pruning_frequency should be greater than 0"
    assert prune_config['max_sparsity_ratio_per_op'] >= 0 and prune_config['max_sparsity_ratio_per_op'] < 1, \
        "pruning_frequency should be greater than 0"
    assert prune_config['pruning_scope'] == "global" or prune_config['pruning_scope'] == "local", \
        "only support 'global' and 'local' prune domain"
    try:
        prune_config['resume_from_pruned_checkpoint'] = bool(prune_config['resume_from_pruned_checkpoint'])
    except:
        assert False, "resume_from_pruned_checkpoint should be bool value"
    if "x" in prune_config["pattern"]:
        pattern = prune_config["pattern"].split('_')[-1].split('x')
        if pattern[0] == "channel" or pattern[1] == "channel":
            pass
        else:
            try:
                N = int(pattern[0])
                M = int(pattern[1])
            except:
                assert False, "N or M can't convert to int"
            assert N > 0, "N should be greater than 0"
            assert M > 0, "M should be greater than 0"
    if ":" in prune_config["pattern"]:
        pattern = prune_config["pattern"].split('_')[-1].split(':')
        try:
            N = int(pattern[0])
            M = int(pattern[1])
        except:
            assert False, "N or M can't convert to int"
        assert N > 0, "N should be greater than 0"
        assert M > N, "M should be greater than N"
        max_ratio = float(N) / M
        assert prune_config['target_sparsity'] <= max_ratio, \
            "in N:M pattern, the max sparsity is N/M={}".format(max_ratio)
        prune_config['max_sparsity_ratio_per_op'] = min(max_ratio, prune_config['max_sparsity_ratio_per_op'])
    if prune_config['reg_coeff'] != None:
        prune_config['reg_coeff'] = float(prune_config['reg_coeff'])
        assert prune_config['reg_coeff'] >= 0, "only support positive reg_type"
    assert prune_config["min_sparsity_ratio_per_op"] >= 0 and prune_config["min_sparsity_ratio_per_op"] <= \
           prune_config['max_sparsity_ratio_per_op'], \
        "min_sparsity_ratio_per_op should in[0, max_sparsity_ratio_per_op]"


def reset_none_to_default(obj, key, default):
    """Functions that add up undefined configurations.

    If some configurations are not defined in the configuration, set it to a default value.

    Args:
        obj: A dict{key: value}
        key: A string. Key in obj.
        default: When the key is not in obj, Add key: default item in original obj.

    """
    if obj == None:
        return None
    if isinstance(obj, dict):
        if (not key in obj.keys()) or obj[key] == None:
            return default
        else:
            return obj[key]
    else:
        if not hasattr(obj, key) or getattr(obj, key) == None:
            return default
        else:
            return getattr(obj, key)


def update_params(info):
    if "parameters" in info.keys():
        params = info["parameters"]
        for key in params:
            info[key] = params[key]


def process_and_check_weight_config(val: WeightPruningConfig):
    default_global_config = {'target_sparsity': 0.9, 'pruning_type': 'snip_momentum', 'pattern': '4x1', 'op_names': [],
                             'excluded_op_names': [],
                             'start_step': 0, 'end_step': 0, 'pruning_scope': 'global', 'pruning_frequency': 1,
                             'min_sparsity_ratio_per_op': 0.0, 'max_sparsity_ratio_per_op': 0.98,
                             'sparsity_decay_type': 'exp',
                             'pruning_op_types': ['Conv', 'Linear'],

                             }
    default_local_config = {'resume_from_pruned_checkpoint': False, 'reg_type': None,
                            'criterion_reduce_type': "mean", 'parameters': {"reg_coeff": 0.0}}

    params_default_config = {"reg_coeff": 0.0}

    default_config = {}
    default_config.update(default_global_config)
    default_config.update(default_local_config)
    default_config.update(params_default_config)

    pruning_configs = val.pruning_configs
    pruners_info = []
    global_info = val.weight_compression
    if len(pruning_configs) == 0:  ##only one
        pruner_info = global_info
        for key in default_config.keys():
            pruner_info[key] = reset_none_to_default(pruner_info, key, default_config[key])
        update_params(pruner_info)
        check_config(pruner_info)
        pruner_info = DotDict(pruner_info)
        pruners_info.append(pruner_info)

    else:  ##TODO need update, in this mode, we ingore the global op names
        for pruner_info in pruning_configs:
            for key in default_config.keys():
                pruner_info[key] = reset_none_to_default(pruner_info, key, global_info[key])
                pruner_info[key] = reset_none_to_default(pruner_info, key, default_config[key])
            update_params(pruner_info)
            check_config(pruner_info)
            pruner_info = DotDict(pruner_info)
            pruners_info.append(pruner_info)

    return pruners_info


def process_config(config):
    """Obtain a config dict object from a config file.
    
    Args:
        config: A string. The path to configuration file.
        
    Returns:
        A config dict object.
    """
    if isinstance(config, WeightPruningConfig):
        return process_and_check_weight_config(config)
    else:
        assert False, f"not supported type {config}"


def parse_to_prune(config, model):
    """Keep target pruned layers."""
    modules = {}
    if config["op_names"] == None or config["op_names"] == []:
        config["op_names"] = [".*"]
    for raw in config["op_names"]:
        try:
            pattern = re.compile(raw)
        except:
            assert False, f"regular expression match does not support {raw}"
        for name, module in filter(lambda t: pattern.search(t[0]), model.named_modules()):
            for layer_type in config["pruning_op_types"]:
                if layer_type in type(module).__name__:
                    modules[name] = module
                    break
    ##remove not to prune layers
    """Drop non-pruned layers."""
    exclude_names = config["excluded_op_names"]
    patterns = [re.compile(s) for s in exclude_names]
    if len(patterns) <= 0:
        return modules
    new_modules = {}
    for name in modules.keys():
        if any([p.search(name) for p in patterns]):
            continue
        new_modules[name] = modules[name]
    return new_modules