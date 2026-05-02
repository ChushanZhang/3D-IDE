"""
Multi-view consistency losses for depth prediction.
Built on the existing GT world_coords (already aligned via axis_align_matrix).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_smoothness_loss(pred_depths, valid_masks=None, gamma=0.5):
    """
    Temporal smoothness: depth between adjacent frames should change smoothly.

    Args:
        pred_depths: (B, S, H, W, 1) predicted depth maps.
        valid_masks: (B, S, H, W) valid-pixel mask.
        gamma: smoothness weight.

    Returns:
        loss: scalar tensor
    """
    B, S, H, W, _ = pred_depths.shape

    if S < 2:
        return torch.zeros((), device=pred_depths.device, dtype=pred_depths.dtype)

    # Depth difference between adjacent frames.
    depth_diff = pred_depths[:, 1:] - pred_depths[:, :-1]  # (B, S-1, H, W, 1)

    # If a mask is provided, only count valid regions.
    if valid_masks is not None:
        # Pixels valid in both frames.
        valid = valid_masks[:, 1:] & valid_masks[:, :-1]  # (B, S-1, H, W)
        valid = valid.unsqueeze(-1)  # (B, S-1, H, W, 1)

        if valid.sum() < 100:
            return torch.zeros((), device=pred_depths.device, dtype=pred_depths.dtype, requires_grad=True)

        loss = (depth_diff[valid] ** 2).mean()
    else:
        loss = (depth_diff ** 2).mean()

    return gamma * loss


def world_coord_consistency_loss(pred_depths, gt_world_coords, gt_depths,
                                  valid_masks=None, gamma=1.0, neighbor_frames=1,
                                  distance_threshold=0.05):
    """
    Multi-view consistency based on GT world_coords.

    Idea:
    1. GT world_coords are in a unified coordinate frame (axis-aligned).
    2. In the overlap region between adjacent frames, points with close
       world_coords correspond to the same 3D location.
    3. Predicted depths at the same 3D point should be consistent (modulo
       the camera-pose difference).

    Args:
        pred_depths: (B, S, H, W, 1) predicted depth.
        gt_world_coords: (B, S, H, W, 3) GT world coords (already aligned).
        gt_depths: (B, S, H, W) GT depth (used as reference).
        valid_masks: (B, S, H, W) valid-pixel mask.
        gamma: loss weight.
        neighbor_frames: how many neighbor frames to compare against
            (1 means only one frame on either side).
        distance_threshold: world-coord distance (meters) below which two
            pixels are treated as the same point.

    Returns:
        loss: scalar tensor
    """
    B, S, H, W, _ = pred_depths.shape

    if S < 2:
        return torch.zeros((), device=pred_depths.device, dtype=pred_depths.dtype)

    total_loss = 0
    num_pairs = 0

    # For every frame, compare with the next `neighbor_frames` frames.
    for i in range(S):
        for offset in range(1, neighbor_frames + 1):
            j = i + offset
            if j >= S:
                continue

            # Frame i and frame j data.
            world_i = gt_world_coords[:, i]  # (B, H, W, 3)
            world_j = gt_world_coords[:, j]  # (B, H, W, 3)
            depth_i = pred_depths[:, i, :, :, 0]  # (B, H, W)
            depth_j = pred_depths[:, j, :, :, 0]  # (B, H, W)
            gt_depth_i = gt_depths[:, i]  # (B, H, W)
            gt_depth_j = gt_depths[:, j]  # (B, H, W)

            # Valid masks.
            if valid_masks is not None:
                valid_i = valid_masks[:, i]  # (B, H, W)
                valid_j = valid_masks[:, j]
            else:
                valid_i = torch.ones(B, H, W, device=pred_depths.device).bool()
                valid_j = torch.ones(B, H, W, device=pred_depths.device).bool()

            # For each pixel in frame i, find the matching pixel in frame j
            # by nearest world_coords.

            # Reshape for efficient computation
            world_i_flat = world_i.reshape(B, H*W, 3)  # (B, H*W, 3)
            world_j_flat = world_j.reshape(B, H*W, 3)  # (B, H*W, 3)

            # Distance matrix: every i pixel to every j pixel.
            # Subsample to keep this affordable.
            sample_ratio = 0.1  # take 10% of pixels
            num_samples = max(int(H * W * sample_ratio), 100)

            # Random sample of frame-i pixels.
            indices_i = torch.randperm(H*W)[:num_samples].to(pred_depths.device)

            sampled_world_i = world_i_flat[:, indices_i, :]  # (B, num_samples, 3)
            sampled_depth_i = depth_i.reshape(B, H*W)[:, indices_i]  # (B, num_samples)
            sampled_gt_depth_i = gt_depth_i.reshape(B, H*W)[:, indices_i]
            sampled_valid_i = valid_i.reshape(B, H*W)[:, indices_i]

            # For each sampled point, find the nearest pixel in frame j.
            # Broadcast distance: (B, num_samples, 1, 3) - (B, 1, H*W, 3).
            distances = torch.norm(
                sampled_world_i.unsqueeze(2) - world_j_flat.unsqueeze(1),
                dim=-1
            )  # (B, num_samples, H*W)

            # Pick the nearest point.
            min_distances, min_indices = distances.min(dim=2)  # (B, num_samples)

            # Keep only matches under the distance threshold.
            valid_matches = (min_distances < distance_threshold) & sampled_valid_i

            if valid_matches.sum() < 10:
                continue

            # Pull the depth at the matched pixel in frame j.
            batch_indices = torch.arange(B, device=pred_depths.device).unsqueeze(1).expand(-1, num_samples)
            matched_depth_j = depth_j.reshape(B, H*W)[batch_indices, min_indices]  # (B, num_samples)
            matched_gt_depth_j = gt_depth_j.reshape(B, H*W)[batch_indices, min_indices]
            matched_valid_j = valid_j.reshape(B, H*W)[batch_indices, min_indices]

            # Final valid mask.
            final_valid = valid_matches & matched_valid_j

            if final_valid.sum() < 10:
                continue

            # Compare depth ratios.
            # Raw depth values differ across cameras due to pose, so we
            # compare predicted depth ratio against GT depth ratio.
            pred_depth_ratio = sampled_depth_i[final_valid] / (matched_depth_j[final_valid] + 1e-6)
            gt_depth_ratio = sampled_gt_depth_i[final_valid] / (matched_gt_depth_j[final_valid] + 1e-6)

            # Ratios should agree.
            loss_pair = torch.abs(pred_depth_ratio - gt_depth_ratio).mean()

            total_loss += loss_pair
            num_pairs += 1

    if num_pairs == 0:
        return torch.zeros((), device=pred_depths.device, dtype=pred_depths.dtype, requires_grad=True)

    return gamma * total_loss / num_pairs


def compute_multiview_depth_loss(predictions, batch,
                                  use_temporal=True, temporal_gamma=0.5,
                                  use_world_consistency=False, world_gamma=0.5,
                                  **kwargs):
    """
    Unified entry point for multi-view depth consistency losses.

    Args:
        predictions: Dict containing 'depth' (B, S, H, W, 1) and 'depth_conf' (B, S, H, W)
        batch: Dict containing 'depth' (GT), 'world_points' (GT world coords), 'point_masks'
        use_temporal: enable the temporal smoothness term.
        temporal_gamma: weight for the temporal smoothness term.
        use_world_consistency: enable the world_coords consistency term.
        world_gamma: weight for the world_coords consistency term.

    Returns:
        loss_dict: dict of losses
    """
    pred_depth = predictions['depth']  # (B, S, H, W, 1)
    gt_depth = batch['depth'].detach()  # (B, S, H, W)
    gt_world_coords = batch.get('world_points', None)  # (B, S, H, W, 3)
    valid_masks = batch.get('point_masks', None)  # (B, S, H, W)

    loss_dict = {}
    total_multiview_loss = 0

    # 1. Temporal smoothness term.
    if use_temporal:
        temporal_loss = temporal_smoothness_loss(
            pred_depth, valid_masks, gamma=temporal_gamma
        )
        loss_dict['temporal_smooth'] = temporal_loss
        total_multiview_loss += temporal_loss

    # 2. World-coords consistency term.
    if use_world_consistency and gt_world_coords is not None:
        world_loss = world_coord_consistency_loss(
            pred_depth, gt_world_coords, gt_depth,
            valid_masks, gamma=world_gamma,
            neighbor_frames=kwargs.get('neighbor_frames', 1),
            distance_threshold=kwargs.get('distance_threshold', 0.05)
        )
        loss_dict['world_consistency'] = world_loss
        total_multiview_loss += world_loss

    loss_dict['total_multiview_loss'] = total_multiview_loss

    return loss_dict
