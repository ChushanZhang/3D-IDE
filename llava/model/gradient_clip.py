# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Union, Optional


class GradientClipper:
    """
    Gradient clipping utils that works for both FSDP and DDP with support for different
    clipping configurations for different parts of the model.
    """
    def __init__(self, configs, accelerator=None, *args, **kwargs):
        """
        Args:
            configs: List of dictionaries, each containing:
                - module_name: str or list of str, module names to apply clipping to
                - max_norm: float, maximum norm for gradient clipping
                - norm_type: int, type of norm (default: 2)
            accelerator: Optional Accelerate accelerator for DeepSpeed compatibility
        """
        self.configs = []
        self.params_to_clip_by_config = None
        self.is_initialized = False
        self.accelerator = accelerator

        for config in configs:
            module_names = config['module_name']
            if isinstance(module_names, str):
                module_names = [module_names]

            self.configs.append({
                'module_names': module_names,
                'max_norm': float(config['max_norm']) if config['max_norm'] is not None else None,
                'norm_type': config.get('norm_type', 2)
            })

    def setup_clipping(self, model: nn.Module, allow_remaining_params: bool = True) -> None:
        """
        Set up gradient clipping by finding all parameters that should be clipped
        based on module names and validating that all parameters are covered.

        This should be called once at the beginning of training.

        Args:
            model: The model to set up gradient clipping for
            allow_remaining_params: If True, allows parameters not in any config to remain unclipped.
                                   If False, raises an error when such parameters are found.
        """
        # First, collect all parameters that should be clipped based on module names
        params_to_clip_by_config = []
        all_clipped_params = set()

        for config in self.configs:
            current_config_params = []
            matched_param_names = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    for module_name in config['module_names']:
                        if module_name in name:
                            current_config_params.append(param)
                            all_clipped_params.add(param)
                            matched_param_names.append(name)
                            break
            params_to_clip_by_config.append((config, current_config_params))
            if len(matched_param_names) > 0:
                print(f"[GradientClipper] Found {len(current_config_params)} params for {','.join(config['module_names'])} (max_norm={config['max_norm']})")
            else:
                print(f"[GradientClipper] No params found for {','.join(config['module_names'])}")

        # Check for remaining parameters
        remaining_params = []
        remaining_param_names = []
        for name, param in model.named_parameters():
            if param.requires_grad and param not in all_clipped_params:
                remaining_params.append(param)
                remaining_param_names.append(name)

        if len(remaining_params) > 0 and not allow_remaining_params:
            print(f"[GradientClipper] ERROR: {len(remaining_params)} params not configured")
            raise ValueError("Some parameters are not configured for gradient clipping")

        # Store the computed parameters
        self.params_to_clip_by_config = params_to_clip_by_config
        self.is_initialized = True

    def __call__(self, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Perform gradient clipping using the pre-computed parameter groups.

        Args:
            model: The model (not used, kept for backward compatibility)

        Returns:
            Dictionary of gradient norms for each configuration
        """
        if not self.is_initialized:
            raise RuntimeError("GradientClipper must be initialized with setup_clipping() before use")

        grad_norms = {}
        for config, params_to_clip in self.params_to_clip_by_config:
            if not params_to_clip or config['max_norm'] is None:
                continue

            # Filter out params without gradients
            # NOTE: In DeepSpeed ZeRO-3, gradients are partitioned across GPUs
            # We need to check if this is a DeepSpeed model
            params_with_grad = []
            for p in params_to_clip:
                # For DeepSpeed ZeRO-3, check if parameter has ds_id (indicates it's managed by DeepSpeed)
                if hasattr(p, 'ds_id'):
                    # This is a DeepSpeed parameter, we need to handle it specially
                    # For now, add it anyway - accelerator.clip_grad_norm_ will handle it
                    params_with_grad.append(p)
                elif p.grad is not None:
                    params_with_grad.append(p)

            if not params_with_grad:
                print(f"[GradientClipper] WARNING: No gradients found for {config['module_names']}")
                print(f"[GradientClipper] Total params to clip: {len(params_to_clip)}, params with grad: 0")
                print(f"[GradientClipper] First param has ds_id: {hasattr(params_to_clip[0], 'ds_id') if params_to_clip else 'N/A'}")
                continue

            # Clip gradients - use torch.nn.utils directly
            # NOTE: For DeepSpeed ZeRO-3, this will work because we're passing
            # DeepSpeed-managed parameters, and PyTorch's clip_grad_norm_ can handle them
            try:
                grad_norm = nn.utils.clip_grad_norm_(
                    params_with_grad,
                    max_norm=config['max_norm'],
                    norm_type=config['norm_type']
                )
            except Exception as e:
                print(f"[GradientClipper] ERROR clipping gradients for {config['module_names']}: {e}")
                continue

            if grad_norm is None or (hasattr(grad_norm, 'item') and (torch.isnan(grad_norm) or torch.isinf(grad_norm))):
                print(f"[GradientClipper] WARNING: grad_norm is None or invalid for {config['module_names']}")
                continue

            grad_norms[",".join(config['module_names'])] = grad_norm.item()

        return grad_norms
