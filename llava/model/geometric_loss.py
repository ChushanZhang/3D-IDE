# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional

from dataclasses import dataclass
# from vggt.utils.pose_enc import extri_intri_to_pose_encoding
# from train_utils.general import check_and_fix_inf_nan
from math import ceil, floor
import code
from llava.utils import rank0_print

def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Defaults to 100.

    Note: Modified for torch.compile compatibility - removes dynamic branching.
    """
    if input_tensor is None:
        return input_tensor

    # ✅ Fix: Remove .any() conditional - always apply torch.where
    # This ensures consistent execution path for torch.compile
    input_tensor = torch.where(
        torch.isnan(input_tensor) | torch.isinf(input_tensor),
        torch.zeros_like(input_tensor),
        input_tensor
    )

    # Apply hard clamping if specified (hard_max is compile-time constant)
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor

@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.

    Supports:
    - Camera loss
    - Depth loss
    - Point loss
    - Coordinate supervision loss (auxiliary loss for coordinate projection)
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, coord=None, track=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        # self.camera = camera
        # self.depth = depth
        # self.point = point
        # self.coord = coord
        # self.track = track
        self.depth = depth or {}
        self.camera = camera or {}
        self.point = point or {}
        self.coord = coord or {}
        self.track = track or {}

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            # 🆕 Include cross-view losses in total depth loss
            depth_loss = (depth_loss_dict["loss_conf_depth"] +
                         depth_loss_dict["loss_reg_depth"] +
                         depth_loss_dict["loss_grad_depth"] +
                         depth_loss_dict["loss_cross_view_depth"] +
                         depth_loss_dict["loss_cross_view_pred"])  # 🆕 Prediction loss
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)
            loss_dict.update({'total_depth_loss': depth_loss})

        # 3D point reconstruction loss - if world points are predicted
        if "world_points" in predictions:
            point_loss_dict = compute_point_loss(predictions, batch, **self.point)
            point_loss = point_loss_dict["loss_conf_point"] + point_loss_dict["loss_reg_point"] + point_loss_dict["loss_grad_point"]
            point_loss = point_loss * self.point["weight"]
            total_loss = total_loss + point_loss
            loss_dict.update(point_loss_dict)
            loss_dict.update({'total_point_loss': point_loss})

        # Coordinate supervision loss is now handled directly in llava_qwen.py
        # (aligned with distill_loss pattern, avoiding OOM from tensor list accumulation)

        # Tracking loss - not cleaned yet, dirty code is at the bottom of this file
        if "track" in predictions:
            raise NotImplementedError("Track loss is not cleaned up yet")

        loss_dict["objective"] = total_loss
        
        # code.interact(local=dict(globals(), **locals()))

        return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['point_masks']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        if valid_frame_mask.sum() == 0:
            # If no valid frames, use efficient zero tensors to avoid gradient issues
            zero_loss = torch.zeros((), device=pred_pose_stage.device, dtype=pred_pose_stage.dtype, requires_grad=True)
            loss_T_stage = zero_loss
            loss_R_stage = zero_loss
            loss_FL_stage = zero_loss
        else:
            # Only consider valid frames for loss computation
            loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
                pred_pose_stage[valid_frame_mask],
                gt_pose_encoding[valid_frame_mask],
                loss_type=loss_type
            )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points'].detach()  # FIX: Detach GT to prevent gradient graph accumulation
    # gt_points_mask = batch['point_masks'] if 'point_masks' in batch else gt_points > 0

    gt_points = check_and_fix_inf_nan(gt_points, "gt_points")

    gt_depth = batch['depth'].detach()  # FIX: Detach GT depth to prevent gradient graph accumulation
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    # gt_depth = gt_depth[..., None]
    gt_points_mask = batch['point_masks'].detach() if 'point_masks' in batch else (gt_depth > 0) & (gt_depth < 10.0)  # FIX: Detach mask too

    # ✅ Early return to avoid expensive computation when insufficient valid points
    # This allows torch.compile to optimize different branches naturally
    num_valid_points = gt_points_mask.sum()
    if num_valid_points < 100:
        zero = torch.zeros((), device=pred_points.device, dtype=pred_points.dtype, requires_grad=True)
        return {
            "loss_conf_point": zero,
            "loss_reg_point": zero,
            "loss_grad_point": zero,
        }

    # Compute regression loss only when we have sufficient valid points
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
        gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range
    )

    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }

    return loss_dict


# geometric_loss.py
def cross_view_depth_consistency_loss(pred_depth, frame_pairs_list, batch, gamma=0.1):
    """
    Cross-view depth consistency loss (torch.compile-friendly).

    Key optimizations:
    1. No Python loops in forward — all per-pair work is precomputed in video_utils.py.
    2. No torch.tensor() / torch.stack() inside forward — uses preallocated tensors.
    3. No dynamic shapes — every tensor's shape is fixed at preprocessing time.
    4. Plays well with torch.compile — no graph breaks, fully compilable.

    This avoids the gradual training-speed regression caused by torch.compile cache blowup.
    """
    if frame_pairs_list is None or len(frame_pairs_list) == 0:
        return torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype, requires_grad=True)

    B, V, pred_H, pred_W, _ = pred_depth.shape
    device = pred_depth.device
    gt_depth = batch['depth'].detach()  # (B, V, H_gt, W_gt)

    # --- 1. Use the preprocessed tensor dict, avoid Python loops in forward ---
    # frame_pairs_list should be a Dict[str, Tensor] produced by video_utils.merge_video_dict.

    # Safety check: if it's not a dict, fall back to zero loss to keep training alive.
    if not isinstance(frame_pairs_list, dict):
        # This should not happen in practice; return zero loss instead of crashing.
        return torch.zeros((), device=device, dtype=pred_depth.dtype, requires_grad=True)

    # Bail out if there are no pairs at all.
    if frame_pairs_list.get('num_pairs', 0) == 0:
        return torch.zeros((), device=device, dtype=pred_depth.dtype, requires_grad=True)

    # Use the prestacked tensors directly (avoid torch.tensor()/torch.stack() in forward).
    batch_idx = frame_pairs_list['batch_indices'].to(device)
    frame_j_idx = frame_pairs_list['frame_j_indices'].to(device)
    valid_mask_all = frame_pairs_list['valid_masks'].to(device)  # already stacked
    weights = frame_pairs_list['weights'].to(device, dtype=pred_depth.dtype)
    num_total_pairs = frame_pairs_list['num_pairs']

    # --- 2. Batched gather of pred / GT depth ---
    pred_depth_j_all = pred_depth[batch_idx, frame_j_idx, :, :, 0]
    gt_depth_j_all = gt_depth[batch_idx, frame_j_idx]

    # --- 3. Batched resize ---
    # valid_mask_all is already prepared above (new dict format / legacy stack path).
    _, H_gt, W_gt = gt_depth_j_all.shape
    _, H_mask, W_mask = valid_mask_all.shape

    # GT preprocessing is wrapped in no_grad to prevent autograd accumulation.
    with torch.no_grad():
        gt_depth_j_all = gt_depth_j_all.unsqueeze(1)
        valid_mask_all = valid_mask_all.unsqueeze(1).float()

        if (H_gt, W_gt) != (pred_H, pred_W):
            gt_depth_j_all = F.interpolate(
                gt_depth_j_all, size=(pred_H, pred_W), mode='bilinear', align_corners=False
            )
        if (H_mask, W_mask) != (pred_H, pred_W):
            valid_mask_all = F.interpolate(
                valid_mask_all, size=(pred_H, pred_W), mode='nearest'
            )

        gt_depth_j_all = gt_depth_j_all.squeeze(1)
        valid_mask_all = valid_mask_all.squeeze(1).bool()

        # --- 4. Batched mask and diff ---
        gt_valid_j = gt_depth_j_all > 0.01  # (N_pairs, H_pred, W_pred)
        final_mask = valid_mask_all & gt_valid_j

    # Use the no-grad final_mask above (already detached from autograd).
    diff = torch.abs(pred_depth_j_all - gt_depth_j_all)
    diff = diff * final_mask.to(diff.dtype)

    # (N_pairs,) — number of valid pixels per pair.
    # Stats use a detached mask.
    with torch.no_grad():
        num_valid_pixels_per_pair = final_mask.sum(dim=[1, 2])
        # (N_pairs,) bool — whether the pair has enough valid pixels.
        valid_pair_mask = (num_valid_pixels_per_pair >= 10)
        safe_num_valid = num_valid_pixels_per_pair.clamp(min=1e-6).float()

    # (N_pairs,) — per-pair mean loss.
    loss_per_pair = diff.sum(dim=[1, 2]) / safe_num_valid

    # --- 5. Drop pairs with < 10 valid pixels and apply per-pair weights ---
    # Zero out the loss contribution of invalid pairs.
    loss_per_pair = loss_per_pair * valid_pair_mask.to(loss_per_pair.dtype)

    # Apply pair weights (only the valid pairs survive).
    loss_per_pair = loss_per_pair * weights

    # Guard against NaN/Inf.
    loss_per_pair = check_and_fix_inf_nan(loss_per_pair, "cross_view_depth_diff")

    # --- 6. Final reduction ---
    total_loss = loss_per_pair.sum()

    # Denominator is the count of valid pairs (matches the original implementation).
    with torch.no_grad():
        num_valid_pairs = valid_pair_mask.sum().clamp(min=1.0).float()

    avg_loss = gamma * total_loss / num_valid_pairs

    # Avoid .item() inside forward (causes GPU sync); summary printing is centralized
    # in qwen2/modeling_qwen2.py.

    return avg_loss


def compute_cross_view_prediction_loss(
    cross_view_predictions: Dict[str, torch.Tensor],
    frame_pairs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.1,
) -> torch.Tensor:
    """
    Cross-view depth PREDICTION loss (not consistency loss).

    This computes the loss between:
    - pred_depth_j_from_i: depth predicted using frame_i features + pose_enc(i→j)
    - gt_depth_j: ground truth depth at frame_j

    Unlike cross_view_depth_consistency_loss which compares frame_j's own prediction
    with gt_depth_j, this function compares the cross-view prediction (from frame_i)
    with gt_depth_j.

    Args:
        cross_view_predictions: Dict from forward_cross_view_depth() containing:
            - 'cross_view_depth': (N_pairs, H, W, 1) predicted depth at frame_j viewpoint
            - 'cross_view_depth_conf': (N_pairs, H, W) confidence
        frame_pairs: Dict containing:
            - 'num_pairs': int
            - 'batch_indices': (N_pairs,) tensor
            - 'frame_j_indices': (N_pairs,) tensor
            - 'valid_masks': (N_pairs, H, W) tensor - pixels visible from both views
            - 'weights': (N_pairs,) tensor - per-pair weights
        batch: Dict containing:
            - 'depth': (B, V, H, W) ground truth depth
        gamma: Weight for the loss

    Returns:
        Scalar loss tensor with gradient
    """
    if cross_view_predictions is None:
        # Return zero loss that maintains gradient flow
        device = batch['depth'].device
        dtype = batch['depth'].dtype
        return torch.zeros((), device=device, dtype=dtype, requires_grad=True)

    if frame_pairs is None or frame_pairs.get('num_pairs', 0) == 0:
        device = batch['depth'].device
        dtype = batch['depth'].dtype
        return torch.zeros((), device=device, dtype=dtype, requires_grad=True)

    # Get predictions
    pred_cross_depth = cross_view_predictions['cross_view_depth']  # (N_valid, H, W, 1)
    pred_H, pred_W = pred_cross_depth.shape[1], pred_cross_depth.shape[2]
    device = pred_cross_depth.device
    dtype = pred_cross_depth.dtype

    # If forward_cross_view_depth filtered pairs, use valid_pair_indices to align
    valid_pair_indices = cross_view_predictions.get('valid_pair_indices', None)

    # Get frame_pairs info, indexed by valid_pair_indices if available
    if valid_pair_indices is not None:
        idx = valid_pair_indices.to(device)
        batch_idx = frame_pairs['batch_indices'].to(device)[idx]
        frame_j_idx = frame_pairs['frame_j_indices'].to(device)[idx]
        valid_mask_all = frame_pairs['valid_masks'].to(device)[idx]
        weights = frame_pairs['weights'].to(device, dtype=dtype)[idx]
    else:
        batch_idx = frame_pairs['batch_indices'].to(device)
        frame_j_idx = frame_pairs['frame_j_indices'].to(device)
        valid_mask_all = frame_pairs['valid_masks'].to(device)
        weights = frame_pairs['weights'].to(device, dtype=dtype)

    # Get GT depth at frame_j
    gt_depth = batch['depth'].detach()  # (B, V, H_gt, W_gt)
    gt_depth_j_all = gt_depth[batch_idx, frame_j_idx]  # (N_pairs, H_gt, W_gt)

    # Resize GT depth and valid_mask to match prediction size
    _, H_gt, W_gt = gt_depth_j_all.shape
    _, H_mask, W_mask = valid_mask_all.shape

    with torch.no_grad():
        gt_depth_j_all = gt_depth_j_all.unsqueeze(1)
        valid_mask_all = valid_mask_all.unsqueeze(1).float()

        if (H_gt, W_gt) != (pred_H, pred_W):
            gt_depth_j_all = F.interpolate(
                gt_depth_j_all, size=(pred_H, pred_W), mode='bilinear', align_corners=False
            )
        if (H_mask, W_mask) != (pred_H, pred_W):
            valid_mask_all = F.interpolate(
                valid_mask_all, size=(pred_H, pred_W), mode='nearest'
            )

        gt_depth_j_all = gt_depth_j_all.squeeze(1)  # (N_pairs, H, W)
        valid_mask_all = valid_mask_all.squeeze(1).bool()

        # Create final mask: valid overlap region AND valid GT depth
        gt_valid_j = gt_depth_j_all > 0.01
        final_mask = valid_mask_all & gt_valid_j

    # Compute L1 loss: |pred_depth_j_from_i - gt_depth_j|
    pred_cross_depth_squeezed = pred_cross_depth[..., 0]  # (N_pairs, H, W)
    diff = torch.abs(pred_cross_depth_squeezed - gt_depth_j_all)
    diff = diff * final_mask.to(diff.dtype)

    # Compute per-pair loss
    with torch.no_grad():
        num_valid_pixels_per_pair = final_mask.sum(dim=[1, 2])
        valid_pair_mask = (num_valid_pixels_per_pair >= 10)
        safe_num_valid = num_valid_pixels_per_pair.clamp(min=1e-6).float()

    loss_per_pair = diff.sum(dim=[1, 2]) / safe_num_valid

    # Filter invalid pairs and apply weights
    loss_per_pair = loss_per_pair * valid_pair_mask.to(loss_per_pair.dtype)
    loss_per_pair = loss_per_pair * weights

    # Check NaN/Inf
    loss_per_pair = check_and_fix_inf_nan(loss_per_pair, "cross_view_prediction_loss")

    # Compute average loss
    total_loss = loss_per_pair.sum()
    with torch.no_grad():
        num_valid_pairs = valid_pair_mask.sum().clamp(min=1.0).float()

    avg_loss = gamma * total_loss / num_valid_pairs

    return avg_loss



def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1,
                       cross_view_gamma=0.1, cross_view_predictions: Optional[Dict[str, torch.Tensor]] = None,
                       cross_view_pred_gamma=0.1, **kwargs):
    """
    Compute depth loss.

    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
            Can also contain 'cross_view_predictions' for prediction loss
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
        cross_view_gamma: Weight for cross-view consistency loss (auto-enabled if frame_pairs exist)
        cross_view_predictions: Optional Dict from forward_cross_view_depth() for prediction loss
            (can also be passed via batch['cross_view_predictions'])
        cross_view_pred_gamma: Weight for cross-view prediction loss
    """
    pred_depth = predictions['depth']               # pred.dim --->  5
    pred_depth_conf = predictions['depth_conf']     # pred.dim --->  4

    # 🆕 Support cross_view_predictions from batch (for compatibility with MultitaskLoss)
    if cross_view_predictions is None:
        cross_view_predictions = batch.get('cross_view_predictions', None)

    gt_depth = batch['depth'].detach()  # FIX: Detach GT to prevent gradient graph accumulation
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    # Skip a redundant clone (read-only here).
    gt_depth_mask = batch['point_masks'].detach() if 'point_masks' in batch else gt_depth > 0

    gt_depth = gt_depth[..., None]              # (B, S, H, W, 1)


    num_valid_depth = gt_depth_mask.sum()
    if num_valid_depth < 100:
        zero = torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype, requires_grad=True)
        return {
            "loss_conf_depth": zero,
            "loss_reg_depth": zero,
            "loss_grad_depth": zero,
            "loss_cross_view_depth": zero,
            "loss_cross_view_pred": zero,  # cross-view prediction loss
        }

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement

    # Compute regression loss only when we have sufficient valid depth data
    loss_conf, loss_grad, loss_reg = regression_loss(
        pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
        gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range
    )

    # Cross-view depth consistency loss.
    # Auto-enabled whenever frame_pairs is present in the batch.
    loss_cross_view = torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype, requires_grad=True)

    if 'frame_pairs' in batch and batch['frame_pairs'] is not None:
        frame_pairs_list = batch['frame_pairs']
        loss_cross_view = cross_view_depth_consistency_loss(
            pred_depth,
            frame_pairs_list,
            batch,
            gamma=cross_view_gamma
        )

    # 🆕 Cross-view depth PREDICTION loss (from frame_i features → predict frame_j depth)
    loss_cross_view_pred = torch.zeros((), device=pred_depth.device, dtype=pred_depth.dtype, requires_grad=True)

    if cross_view_predictions is not None and 'frame_pairs' in batch and batch['frame_pairs'] is not None:
        loss_cross_view_pred = compute_cross_view_prediction_loss(
            cross_view_predictions,
            batch['frame_pairs'],
            batch,
            gamma=cross_view_pred_gamma
        )

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,
        f"loss_grad_depth": loss_grad,
        f"loss_cross_view_depth": loss_cross_view,  # Consistency loss
        f"loss_cross_view_pred": loss_cross_view_pred,  # 🆕 Prediction loss
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape


    # depth loss
        # mask.shape >>> torch.Size([1, S, 378, 378])
        # gt.shape >>> torch.Size([1, S, 378, 378, 1])
        # pred.shape >>> torch.Size([1, S, 378, 378, 1])

    # point loss
        # mask.shape >>> torch.Size([1, S, 378, 378])
        # gt.shape >>> torch.Size([1, S, 378, 378, 3])
        # pred.shape >>> torch.Size([1, S, 378, 378, 3])


    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    # loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    # Fix: clamp conf before log() to avoid -inf when conf hits zero.
    safe_conf = torch.clamp(conf[mask], min=1e-6)  # critical: clamp before log
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(safe_conf)
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0
    

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # ✅ Fix: Process confidence-weighted loss without dynamic branching
    # Always execute same path for torch.compile compatibility
    has_conf_elements = loss_conf.numel() > 0
    if has_conf_elements:
        # Filter out outliers using quantile-based thresholding
        if valid_range > 0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        # Use efficient zero tensor to maintain gradient flow
        loss_conf = torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)

    # ✅ Fix: Process regular regression loss without dynamic branching
    has_reg_elements = loss_reg.numel() > 0
    if has_reg_elements:
        # Filter out outliers using quantile-based thresholding
        if valid_range > 0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        # Use efficient zero tensor to maintain gradient flow
        loss_reg = torch.zeros((), device=pred.device, dtype=pred.dtype, requires_grad=True)

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # ✅ Fix: Constant-shape path for torch.compile compatibility
    # Keep computation on original grid shape (4, B, H, W) - no dynamic reshaping

    # Compute cosine similarity on full grid (maintains constant shape)
    dot = (pred_normals * gt_normals).sum(dim=-1)  # (4, B, H, W)
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta) for numerical stability
    loss = 1.0 - dot
    loss = check_and_fix_inf_nan(loss, "normal_loss")

    # Apply confidence weighting if provided (on full grid)
    if conf is not None:
        conf4 = conf[None, ...].expand_as(loss)  # (4, B, H, W)
        # ✅ Fix: Clamp to avoid log(<=0) which causes NaN/Inf
        conf4 = torch.clamp(conf4, min=1e-6)
        loss = gamma * loss * conf4 - alpha * torch.log(conf4)
        loss = check_and_fix_inf_nan(loss, "normal_loss_with_conf")

    # ✅ Fix: Tensor-based gating with constant shape (no Python branching)
    K_t = all_valid.sum()  # 0-dim tensor
    has_enough = (K_t >= 10)

    # Weight matrix: valid pixels = 1, invalid = 0
    w = all_valid.to(loss.dtype)  # (4, B, H, W)
    # Gate all weights to 0 when K < 10 (using torch.where)
    w = torch.where(has_enough, w, torch.zeros_like(w))

    # Compute weighted mean: K>=10 → mean of valid pixels; K<10 → 0
    num = (loss * w).sum()
    den = torch.clamp(w.sum(), min=1.0)
    result = num / den

    return result


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        # ✅ Fix: Clamp confidence to avoid log(<=0) which causes -inf
        conf_x = torch.clamp(conf_x, min=1e-6)
        conf_y = torch.clamp(conf_y, min=1e-6)

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    # ✅ Fix: Safe division with gating (no Python branching, no actual division by zero)
    # Note: torch.where doesn't short-circuit, so we avoid putting division inside it
    grad_loss_sum = torch.sum(grad_loss)
    has_valid = (divisor > 0)

    # Use safe denominator to prevent actual division by zero
    div_f = divisor.to(grad_loss_sum.dtype)
    safe_div = torch.clamp(div_f, min=1.0)

    # Gate result: divisor==0 → result=0 (via multiplication by 0); divisor>0 → result=sum/divisor
    result = (grad_loss_sum / safe_div) * has_valid.to(grad_loss_sum.dtype)

    return result


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.

    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.

    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization

    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    # Convert to float32 for cross product compatibility
    point_map = point_map.float()

    # Pad inputs to avoid boundary issues
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
    pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

    # Get neighboring points for each pixel
    center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
    up     = pts[:, :-2,  1:-1, :]
    left   = pts[:, 1:-1, :-2 , :]
    down   = pts[:, 2:,   1:-1, :]
    right  = pts[:, 1:-1, 2:,   :]

    # Compute direction vectors from center to neighbors
    up_dir    = up    - center
    left_dir  = left  - center
    down_dir  = down  - center
    right_dir = right - center

    # Compute four cross products for different normal directions
    n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
    n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
    n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
    n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

    # Validity masks - require both direction pixels to be valid
    v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
    v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
    v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
    v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

    # Stack normals and validity masks
    normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
    valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

    # Normalize normal vectors
    normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.

    This helps remove outliers that could destabilize training.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor (same shape as input)

    Note: Modified for torch.compile compatibility:
        - No early returns (consistent execution path)
        - Fixed output shape (always returns original shape)
        - Deterministic sampling (only for computing quantile, no torch.randperm)
        - Preserves numerical semantics via scaling
        - All operations use tensor types (no implicit conversions)
    """
    # 1. Flatten and clamp original tensor
    flat = loss_tensor.view(-1)
    numel = flat.numel()
    flat = torch.clamp(flat, min=-hard_max, max=hard_max)

    # ✅ Fix: Keep numel as tensor to avoid implicit type conversions
    numel_t = torch.tensor(numel, device=flat.device, dtype=torch.long)

    # 2. Create subsample ONLY for computing quantile threshold
    # ✅ Fix: Don't overwrite flat - use separate stats_flat
    max_samples = 1_000_000
    if numel > max_samples:
        # ✅ Fix: Use ceiling division + arange (avoid linspace integer issues)
        step = (numel + max_samples - 1) // max_samples
        idx = torch.arange(0, numel, step, device=flat.device)[:max_samples]
        stats_flat = flat[idx]  # Separate variable for statistics only
    else:
        stats_flat = flat

    # 3. Compute quantile threshold on subsample
    # ✅ Fix: Use torch.clamp instead of Python min() to avoid tensor/scalar mixing
    q = torch_quantile(stats_flat.detach(), valid_range)
    q = torch.clamp(q, max=hard_max)

    # 4. Build mask on ORIGINAL flat (not subsample!)
    # ✅ Fix: mask and scaling happen on original tensor
    mask = (flat < q)
    num_valid = mask.sum()

    # 5. Handle min_elements logic with tensor operations (no branching)
    # ✅ Fix: Preserve min_elements behavior using torch.where
    should_filter = num_valid > min_elements
    effective_mask = torch.where(
        should_filter,
        mask,
        torch.ones_like(mask, dtype=torch.bool)
    )
    effective_num_valid = torch.where(
        should_filter,
        num_valid,
        numel_t
    )

    # 6. Compute scaling factor with tensor operations
    # Scale ensures: (flat * mask * scale).mean() == flat[mask].mean()
    denom = torch.clamp(effective_num_valid, min=1)
    scale = (numel_t.to(flat.dtype) / denom.to(flat.dtype))

    # 7. Apply mask and scaling on original flat
    flat = flat * effective_mask.to(flat.dtype) * scale

    # 8. Restore original shape (always same shape - compile-friendly!)
    return flat.view_as(loss_tensor)


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Convert to float32 if needed (kthvalue doesn't support BFloat16 on CUDA)
    # Note: We keep output in float32 to avoid creating extra tensors that could break gradient flow
    if input.dtype == torch.bfloat16:
        input = input.float()

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

