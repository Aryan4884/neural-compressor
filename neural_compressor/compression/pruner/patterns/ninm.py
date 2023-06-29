"""N:M patterns."""
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
from .base import (register_pattern,
                   PytorchBasePattern,
                   SparsityInfo,
                   ProgressivePatternUtils)
from ..utils import logger
from neural_compressor.utils.utility import LazyImport

torch = LazyImport('torch')
tf = LazyImport('tensorflow')


@register_pattern('ptN:M')
class PytorchPatternNInM(PytorchBasePattern):
    """Pruning Pattern.

    A Pattern class derived from Pattern. In this pattern, N out of every M continuous weights will be pruned.
    For more info of this pattern, please refer to :
    https://github.com/intel/neural-compressor/blob/master/docs/sparsity.md

    Args:
        config: A config dict object that contains the pattern information.

    Attributes:
        N: The number of elements to be pruned in a weight sequence.
        M: The size of the weight sequence.
    """

    def __init__(self, config, modules):
        """Initialize the basic pruning unit of N:M pattern."""
        super(PytorchPatternNInM, self).__init__(config, modules)
        pattern = self.pattern.split('_')[-1]
        self.N = int(pattern.split(':')[0])
        self.M = int(pattern.split(':')[1])  # m is bigger
        self.check_layer_validity(self.modules, (self.N, self.M))

    def check_layer_validity(self, datas: dict, block_size: tuple):
        """Check if a layer is valid for this block_size.

        Args:
            datas: A dict object containing the weights for all layers.
            block_size: A tuple representing the size of the pattern block.
        """
        self.invalid_layers = []
        for key in datas.keys():
            data = datas[key].weight
            data = self._reshape_orig_to_2dims(data)
            shape = data.shape
            if shape[1] % block_size[1] != 0:
                self.invalid_layers.append(key)
                logger.warning(f"{key} shape {shape} cannot be divided by {self.pattern}")

    def get_reduced_masks_from_data(self, data, key):
        """Obtain the unpruned weights and reshape according to the block_size.

        Args:
            data: Input.
            key: The layer name.

        Returns:
            A tensor representing the unpruned weights.
        """
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        M = self.M
        N = self.N
        new_shape = [shape[0], shape[1] // M, M]
        data = data.reshape(new_shape)
        nonzeros = torch.count_nonzero(data, dim=-1)
        reduced_mask = nonzeros > N
        return reduced_mask

    def get_least_ninm_mask_from_data(self, score):
        """Generate the least N scores in M.

        Args:
            score: the pruning scores of weights.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated.
                1 means unpruned and 0 means pruned.
        """
        current_score = score
        M = self.M
        N = self.N
        current_score = self._reshape_orig_to_2dims(current_score)
        shape = current_score.shape
        new_shape = [shape[0], shape[1] // M, M]
        current_score_new = current_score.reshape(new_shape)

        threshold, _ = torch.kthvalue(current_score_new, N, dim=2)
        threshold = threshold.unsqueeze(-1)

        threshold = threshold.expand(shape[0], shape[1] // M, M)
        threshold = threshold.reshape((shape[0], shape[1]))

        one = torch.tensor([1.]).to(current_score.device)
        zero = torch.tensor([0.]).to(current_score.device)
        mask = torch.where(current_score <= threshold, zero, one)
        return mask

    def get_sparsity_ratio(self, pre_masks, return_dict=False):
        """Please note that the zero cnt and total cnt are all block_wise for supporting channel-wise pruning.

        The return sparsity ratio is elementwised.

        Args:
            pre_masks: Dict{"layer_name": Tensor} that stores the masks generated after the last pruning step.
            return_dict: A bool determining whether to return more information like zero_cnt and total_cnt.

        Returns:
            An elementwise sparisty ratio.
        """
        zero_cnt = 0
        total_cnt = 0
        for key in pre_masks.keys():
            if key in self.invalid_layers:
                # total_cnt += pre_masks[key].numel() // self.M
                continue
            reduced_mask = self.get_reduced_masks_from_data(pre_masks[key], key)
            zero_cnt += int((torch.sum(reduced_mask == 0)).data.item())
            total_cnt += int(reduced_mask.numel())
        sparsity_ratio = float(zero_cnt) / total_cnt * self.N / self.M

        if return_dict:
            return {"sparsity_ratio": sparsity_ratio, "zero_cnt": zero_cnt,
                    "total_cnt": total_cnt}
        else:
            return sparsity_ratio

    def _reshape_orig_to_2dims(self, data):
        """Process layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.

        Returns:
            Reshaped data.
        """
        if len(data.shape) == 4:  # TODO: need to verify whether it's ok for transposed conv
            data = data.permute(0, 2, 3, 1)  # cout,k,k,cin
            data = data.reshape(data.shape[0], -1)
        return data

    def _reshape_2dims_to_orig(self, data, orig_shape):
        """Recover layers that are not two-dimensional(e.g conv layer).

        Args:
            data: Input.

        Returns:
            Reshaped data.
        """
        if len(orig_shape) == 4:
            data = data.reshape(orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[1])
            data = data.permute(0, 3, 1, 2)
        return data

    def reshape_orig_to_pattern(self, data, key):
        """Reshape the data based on the pruning pattern.

        Args:
            data: Input.
            key: layer name.

        Returns:
            Reshaped data.
        """
        data = self._reshape_orig_to_2dims(data)
        shape = data.shape
        new_shape = [shape[0], shape[1] // self.M, self.M]
        data = data.reshape(new_shape)
        return data

    def reshape_reduced_to_orig(self, data, key, orig_shape):
        """Reshape the reduced data to its original shape.

        Args:
            data: Input.
            key: The layer name.
            orig_shape: The original shape of the layer.

        Returns:
            Data of its original shape.
        """
        data = data.repeat_interleave(self.M, dim=-1)
        return self._reshape_2dims_to_orig(data, orig_shape)

    def reduce_scores(self, scores):
        """Calculate the pruning scores after reducing the data and obtain the least N scores in M.

        Args:
            scores: Pruning scores of weights.

        Returns:
            Updated pruning scores and the least N scores in M.
        """
        # to get the least N scores in M
        M = self.M
        least_ninm_masks = {}
        new_scores = {}
        for key in scores.keys():
            if key in self.invalid_layers:
                continue
            if self.keep_mask_layers.get(key, False):
                continue
            current_score = scores[key]
            mask = self.get_least_ninm_mask_from_data(current_score)
            current_score_new = self._reshape_orig_to_2dims(current_score)
            shape = current_score_new.shape
            current_score_new = current_score_new.reshape((shape[0], shape[1]))
            # to get the sum of N scores in each block with M
            current_score_new = current_score_new * (1.0 - mask)
            current_score_new = current_score_new.reshape(shape[0], shape[1] // M, M)
            score_sum = self.reduce_tensor(current_score_new, dim=-1)
            least_ninm_masks[key] = mask
            new_scores[key] = score_sum
        return new_scores, least_ninm_masks

    def get_ele_mask_per_threshold(self, score, threshold, block_size, least_ninm_mask):
        """Get the elementwise mask per threshold.

        Args:
            score: A tensor that stores the pruning scores of weights.
            threshold: A float used to determine whether to prune a weight.
            block_size: A list of two integers representing the height and width of the block.
            least_m_in_m_masks: A tensor representing the least N scores in M.

        Returns:
            mask: The elementwise pruning mask.
        """
        zero = torch.tensor([0.]).to(score.device)
        one = torch.tensor([1.]).to(score.device)
        mask = torch.where(score <= threshold, zero, one)
        mask = mask.repeat_interleave(block_size[1], dim=-1)
        # both zero will be zero
        mask = (mask + least_ninm_mask)
        mask = torch.where(mask <= 0, zero, one)
        return mask

    def get_masks_global(self, scores, cur_target_sparsity_ratio, pre_masks,
                         keep_exact_sparsity_ratio=True):
        """Generate masks for layers.

        Gather all layer's scores together and calculate a common threshold.
        This threshold will be applied for all layers.

        Args:
            scores: A dict{"layer_name": Tensor} that stores the pruning scores of weights.
            target_sparsity_ratio: A float representing the model's final sparsity.
            pre_masks: A dict{"layer_name": Tensor} representing the masks generated after the last pruning step.
            max_sparsity_ratio_per_op: A float representing the maximum sparsity that one layer can reach.

        Returns:
            A dict with the identical size as pre_masks and its 0/1 values are updated.
                1 means unpruned and 0 means pruned.
        """
        masks = pre_masks

        block_sparsity_ratio = cur_target_sparsity_ratio * self.M / self.N
        k_blockwise = self.update_residual_cnt(pre_masks, block_sparsity_ratio)
        if k_blockwise <= 0:
            return masks
        new_scores, least_ninm_masks = self.reduce_scores(scores)
        global_scores = torch.cat([torch.flatten(v) for v in new_scores.values()])  # block_wise
        residual_k = k_blockwise
        not_exceed_layers = [key for key in new_scores.keys()]

        while True:
            threshold, _ = torch.kthvalue(global_scores, residual_k)
            for key in not_exceed_layers:
                score = new_scores[key]
                mask = self.get_ele_mask_per_threshold(score, threshold, (self.N, self.M), least_ninm_masks[key])
                info = self.get_sparsity_ratio({key: mask}, return_dict=True)
                zero_cnt = info["zero_cnt"]
                total_cnt = info["total_cnt"]
                current_sparsity_ratio = float(zero_cnt) / total_cnt
                key_new_sparsity = SparsityInfo(zero_cnt, total_cnt, current_sparsity_ratio)
                need_adjust, adjust_ratio = self.adjust_ratio(masks, key, key_new_sparsity,
                                                              self.max_sparsity_ratio_per_op * self.M / self.N,
                                                              self.min_sparsity_ratio_per_op * self.M / self.N,
                                                              self.target_sparsity_ratio * self.M / self.N)

                if need_adjust:
                    self.keep_mask_layers[key] = True
                    masks[key] = self.get_single_mask_per_target_ratio(new_scores[key], adjust_ratio)
                    masks[key] = masks[key].repeat_interleave(self.M, dim=-1)
                    # both zero will be zero
                    masks[key] = (masks[key] + least_ninm_masks[key])
                    zero = torch.tensor([0.]).to(score.device)
                    one = torch.tensor([1.]).to(score.device)
                    masks[key] = torch.where(masks[key] <= 0, zero, one)
                    if keep_exact_sparsity_ratio:
                        zero_cnt = self.get_sparsity_ratio({key: masks[key]}, return_dict=True)["zero_cnt"]
                        residual_k -= zero_cnt
                else:
                    masks[key] = mask
                masks[key] = masks[key].bool()
            if not keep_exact_sparsity_ratio:
                break
            new_not_exceed_layers = [key for key in new_scores.keys() if not self.keep_mask_layers.get(key, False)]
            if not_exceed_layers == new_not_exceed_layers or len(new_not_exceed_layers) == 0:
                break
            not_exceed_layers = new_not_exceed_layers
            global_scores = torch.cat([torch.flatten(new_scores[key]) for key in not_exceed_layers])

        for key in masks.keys():
            if key in self.invalid_layers:
                continue
            if len(scores[key].shape) == 4:  # need to permute
                mask = masks[key]
                orig_shape = scores[key].shape
                mask = self._reshape_2dims_to_orig(mask, orig_shape)
                masks[key] = mask
            zero_cnt = False if self.low_memory_usage else 0.0
            layer_ratio = torch.sum(masks[key] == zero_cnt).data.item() / masks[key].numel()
            logger.info(f'layer {key} sparsity_ratio is {layer_ratio}')
        return masks

    def get_pattern_lock_masks(self, modules):
        """Obtain masks from original weight map, by masking where weights' are zero.

        Args:
            modules: A dict{"layer_name": Tensor} that stores weights.

        Returns:
            A dict with the identical size as modules, containing pattern lock masks.
        """
        pattern_lock_masks = {}
        for key in modules.keys():
            weight = modules[key].weight
            orig_shape = weight.shape
            if key in self.invalid_layers:
                mask = torch.ones(orig_shape, device=weight.device)
                pattern_lock_masks[key] = mask
                continue
            mask = self.get_least_ninm_mask_from_data(weight)
            mask = self._reshape_2dims_to_orig(mask, orig_shape)
            pattern_lock_masks[key] = mask
        return pattern_lock_masks

    def update_progressive_masks(self, pre_masks, cur_masks, scores, progressive_step, progressive_configs):
        assert progressive_configs['progressive_type'] == "scores", "N:M progressive pruning only supports 'scores'."
        # we only have to handle global score or local score
        return ProgressivePatternUtils.update_progressive_masks_scores_order(pre_masks, cur_masks, scores,
                                                                             progressive_step, progressive_configs)