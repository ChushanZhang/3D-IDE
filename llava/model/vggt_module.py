"""
VGGT DPT/DPTHEAD Module for learning geometric information
This module integrates pretrained VGGT heads for depth, point maps, and camera pose prediction.
"""

import code
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union
import os
import types

from llava.utils import rank0_print
import sys
vggt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'vggt')
if vggt_path not in sys.path:
    sys.path.append(vggt_path)
from vggt.heads.dpt_head import DPTHead, custom_interpolate
from vggt.heads.camera_head import CameraHead
from vggt.heads.head_act import activate_head


def safe_get_rank():
    """Safely get rank, returning 0 if DeepSpeed is not initialized."""
    try:
        from deepspeed.comm import get_rank, is_initialized
        if is_initialized():
            return get_rank()
    except (ImportError, AssertionError):
        pass
    return 0




class NoOpDPTModule(nn.Module):
    """
    Placeholder module used when the real dpt_module is not present.
    Accepts any inputs and returns None to match the expected (result, None) structure.
    """
    def __init__(self):
        super().__init__()
        # Keep an empty parameter list so this is a valid nn.Module.

        # coord_supervision_module is also None (not needed at inference).
        self.coord_supervision_module = None

    def forward(self, multi_level_feats, **kwargs):
        # No gradient computation required; just return None.
        # **kwargs absorbs extra args (e.g. frame_pairs) for forward-compatibility.
        return None


def _patched_dpt_head_forward(
    self,
    aggregated_tokens_list: List[torch.Tensor],
    patch_start_idx: int,
    B: int, S: int, H: int, W: int,
    frames_chunk_size: int = 8,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    This is a monkey-patched version of DPTHead.forward.
    It avoids using the `images` tensor and instead uses shape information directly,
    which prevents the creation of a disconnected tensor that causes issues with DeepSpeed.
    """

    def _forward_chunk(
        tokens_list: List[torch.Tensor],
        start_idx: int,
        chunk_b: int, chunk_s: int, chunk_h: int, chunk_w: int,
        frame_start: int = None, frame_end: int = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        patch_h, patch_w = chunk_h // self.patch_size, chunk_w // self.patch_size
        out = []
        dpt_idx = 0
        for layer_idx in self.intermediate_layer_idx:
            x = tokens_list[layer_idx][:, :, start_idx:]
            if frame_start is not None and frame_end is not None:
                x = x[:, frame_start:frame_end]
            x = x.view(chunk_b * chunk_s, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, chunk_w, chunk_h)
            x = self.resize_layers[dpt_idx](x)
            out.append(x)
            dpt_idx += 1
        out = self.scratch_forward(out)
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )
        if self.pos_embed:
            out = self._apply_pos_embed(out, chunk_w, chunk_h)
        if self.feature_only:
            return out.view(chunk_b, chunk_s, *out.shape[1:])
        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        preds = preds.view(chunk_b, chunk_s, *preds.shape[1:])
        conf = conf.view(chunk_b, chunk_s, *conf.shape[1:])
        return preds, conf

    if frames_chunk_size is None or frames_chunk_size >= S:
        return _forward_chunk(aggregated_tokens_list, patch_start_idx, B, S, H, W)

    assert frames_chunk_size > 0

    # Process chunks and concatenate (original working approach)
    if self.feature_only:
        all_preds = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_S = frames_end_idx - frames_start_idx
            chunk_output = _forward_chunk(
                aggregated_tokens_list, patch_start_idx, B, chunk_S, H, W, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_output)
        return torch.cat(all_preds, dim=1)
    else:
        all_preds = []
        all_conf = []
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            chunk_S = frames_end_idx - frames_start_idx
            chunk_preds, chunk_conf = _forward_chunk(
                aggregated_tokens_list, patch_start_idx, B, chunk_S, H, W, frames_start_idx, frames_end_idx
            )
            all_preds.append(chunk_preds)
            all_conf.append(chunk_conf)
        return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)


class VGGTModule(nn.Module):
    """
    Unified wrapper for VGGT heads including depth, point maps, and camera pose prediction.
    This module uses pretrained VGGT weights for auxiliary tasks during training.
    """

    def __init__(
        self,
        vision_hidden_size: int,
        checkpoint_path: Optional[str] = None,
        use_pretrained: bool = True,
        patch_size: int = 14,
        embed_dim: int = 1024,
        intermediate_layer_idx: Optional[List[int]] = None,
        enable_depth: bool = True,
        enable_points: bool = True,
        resolution: int = 384,
        llm_hidden_size: Optional[int] = None,
    ):
        """
        Initialize the VGGT Module.

        Note: Relative pose encoding is automatically enabled when frame_pairs are provided.

        Args:
            llm_hidden_size: LLM hidden dimension for CoordSupervisionModule (e.g., 3584 for Qwen2-7B)
                           If None, coord_supervision_module will not be initialized
        """
        super().__init__()

        self.vision_hidden_size = vision_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.resolution = resolution
        self.use_pretrained = use_pretrained
        self.enable_depth = enable_depth
        self.enable_points = enable_points

        if intermediate_layer_idx is None:
            intermediate_layer_idx = [6, 12, 18, 24]

        self.intermediate_layer_idx = intermediate_layer_idx
        
        if enable_depth:
            self.depth_head = DPTHead(dim_in=2 * self.embed_dim, output_dim=2, activation="exp",
                                        conf_activation="expp1",  patch_size=self.patch_size,
                                        intermediate_layer_idx=self.intermediate_layer_idx)
            self.depth_head.forward = types.MethodType(_patched_dpt_head_forward, self.depth_head)

        if enable_points:
            # self.point_head = DPTHead(dim_in=2 * self.embed_dim, output_dim=4, activation="inv_log",
            #                             conf_activation="expp1",  patch_size=self.patch_size,
            #                             intermediate_layer_idx=self.intermediate_layer_idx)
            self.point_head = DPTHead(dim_in=2 * self.embed_dim, output_dim=4, activation="linear",
                                        conf_activation="expp1",  patch_size=self.patch_size,
                                        intermediate_layer_idx=self.intermediate_layer_idx)
            self.point_head.forward = types.MethodType(_patched_dpt_head_forward, self.point_head)


        # Initialize feature_projector first (will be converted to bf16/fp16 by trainer)
        # LayerNorm at the end to normalize features and prevent gradient explosion

        self.camera_pose_embedding = nn.ParameterList(
            [nn.Parameter(torch.zeros(4*4, self.vision_hidden_size)) for _ in range(len(self.intermediate_layer_idx))]
        )

        self.feature_projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, self.vision_hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.vision_hidden_size * 2, 2048),
            nn.LayerNorm(2048)  # Normalize output to stabilize DPT heads
        )

        # Load pretrained weights if specified (after all modules are initialized)
        if use_pretrained:
            self._load_vggt_checkpoint(checkpoint_path)
        else:
            print("use_pretrained=False, Using random initialization for VGGT heads.")

    def _load_vggt_checkpoint(self, checkpoint_path: str):
        """Load VGGT checkpoint and extract relevant head weights."""
        if not os.path.exists(checkpoint_path):
            print(f"Warning: VGGT checkpoint path '{checkpoint_path}' does not exist. Using random initialization.")
            exit()
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            state_dict = None
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and any(k.startswith(('depth_head', 'point_head',)) for k in checkpoint.keys()):
                state_dict = checkpoint
            else:
                state_dict = checkpoint

            if state_dict is None:
                print(f"Warning: Could not find compatible state dict in checkpoint: {checkpoint_path}")
                return

            if self.enable_depth and hasattr(self, 'depth_head'):
                depth_state = {k.replace('depth_head.', ''): v
                             for k, v in state_dict.items()
                             if k.startswith('depth_head.')}
                if depth_state:
                    try:
                        self.depth_head.load_state_dict(depth_state, strict=False)
                        if safe_get_rank() == 0:
                            print(f"Successfully loaded depth head from VGGT checkpoint")
                            # Print weight statistics for diagnosis
                            total_params = sum(p.numel() for p in self.depth_head.parameters())
                            max_weight = max(p.abs().max().item() for p in self.depth_head.parameters())
                            mean_weight = sum(p.abs().mean().item() * p.numel() for p in self.depth_head.parameters()) / total_params
                            print(f"  Depth head weight stats: max={max_weight:.4f}, mean={mean_weight:.6f}, total_params={total_params}")
                    except Exception as e:
                        if safe_get_rank() == 0:
                            print(f"Warning: Failed to load depth head: {e}")

            if self.enable_points and hasattr(self, 'point_head'):
                point_state = {k.replace('point_head.', ''): v
                             for k, v in state_dict.items()
                             if k.startswith('point_head.')}
                if point_state:
                    try:
                        self.point_head.load_state_dict(point_state, strict=False)
                        print(f"Successfully loaded point head from VGGT checkpoint")
                    except Exception as e:
                        print(f"Warning: Failed to load point head: {e}")

            if hasattr(self, 'camera_head') and self.enable_camera:
                camera_state = {k.replace('camera_head.', ''): v
                              for k, v in state_dict.items()
                              if k.startswith('camera_head.')}
                if camera_state:
                    try:
                        self.camera_head.load_state_dict(camera_state, strict=False)
                        print(f"Successfully loaded camera head from VGGT checkpoint")
                    except Exception as e:
                        print(f"Warning: Failed to load camera head: {e}")


        except Exception as e:
            print(f"Warning: Failed to load VGGT checkpoint from '{checkpoint_path}': {e}")
            print("Wrong ckpt_path, Using random initialization for VGGT heads.")

    def prepare_vggt_features(
        self,
        multilevel_features: List[torch.Tensor],
        frame_pairs: Optional[List[List[Dict]]] = None,
        B: int = None,
        S: int = None,
        apply_pose_encoding: bool = True,
    ) -> List[torch.Tensor]:
        """
        Prepare features in VGGT format by projecting to 2*vision_hidden_size dimensions.

        Args:
            multilevel_features: List of (B, S, num_tokens, vision_hidden_size) tensors
            frame_pairs: List[List[Dict]], frame_pairs[b] holds all pairs for batch b
                each pair contains:
                    - frame_i: int, source frame index [0, S)
                    - frame_j: int, target frame index [0, S)
                    - T_relative: (4, 4) tensor
                    - type: str
            B: Batch size
            S: Number of frames per batch
            apply_pose_encoding: bool, whether to apply pose encoding to features.
                - True: Add pose encoding (for cross-view depth prediction)
                - False: No pose encoding (for self depth prediction)

        Returns:
            aggregated_tokens_list: List of (B, S, num_tokens, 2*vision_hidden_size) tensors
        """
        if multilevel_features is None:
            raise ValueError("multilevel_features cannot be None")

        # Infer B and S from the input shape if not provided.
        if B is None or S is None:
            first_shape = multilevel_features[0].shape
            B = first_shape[0]
            S = first_shape[1]

        max_layer_idx = max(self.intermediate_layer_idx)
        aggregated_tokens_list = [None] * (max_layer_idx + 1)

        if len(multilevel_features) != len(self.intermediate_layer_idx):
            raise ValueError(f"Expected {len(self.intermediate_layer_idx)} features, got {len(multilevel_features)}")

        # Compute the relative pose encoding per frame (only if apply_pose_encoding=True).
        pose_encodings_per_layer = None
        if apply_pose_encoding and frame_pairs is not None:
            # frame_pairs is the dict produced by merge_video_dict.
            # Make sure num_pairs > 0 before trying to use it.
            if isinstance(frame_pairs, dict) and frame_pairs.get('num_pairs', 0) > 0:
                # Pass the flattened tensor dict directly to the pose-encoding helper.
                pose_encodings_per_layer = self._compute_pose_encodings(frame_pairs, B, S)

        # Project and assign in one loop
        for i, (features, layer_idx) in enumerate(zip(multilevel_features, self.intermediate_layer_idx)):
            # features: (B, S, num_tokens, vision_hidden_size)

            if pose_encodings_per_layer is not None:
                features = features + pose_encodings_per_layer[i].unsqueeze(2)

            projected = self.feature_projector(features)
            aggregated_tokens_list[layer_idx] = projected

        return aggregated_tokens_list

    
    def _compute_pose_encodings(
        self,
        frame_pairs_dict: Dict[str, torch.Tensor],
        B: int,
        S: int
    ) -> List[torch.Tensor]:
        """
        Vectorized computation of relative pose encodings for each source frame.
        """
        num_layers = len(self.intermediate_layer_idx)
        device = self.camera_pose_embedding[0].device
        dtype = self.camera_pose_embedding[0].dtype

        # 1. Allocate (L, B, S, D); we assume B=1 so (L, S, D) is enough.
        if B != 1:
            # The vectorized index_add_ logic below assumes B=1; for B > 1
            # the batch_idx handling needs to be added.
            raise NotImplementedError("Vectorized _compute_pose_encodings does not yet support B > 1.")

        pose_encodings = [
            torch.zeros(S, self.vision_hidden_size, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]

        # 2. Short-circuit when there are no pairs.
        if frame_pairs_dict['num_pairs'] == 0:
            return [pe.unsqueeze(0) for pe in pose_encodings]  # restore (B, S, D)

        # 3. Pull the flattened tensors straight out of the dict (avoid restacking).
        # (N_pairs, 4, 4) -> (N_pairs, 16)
        T_rels_flat_batch = frame_pairs_dict['T_relatives'].to(
            device=device, dtype=dtype
        ).view(frame_pairs_dict['num_pairs'], -1)

        # (N_pairs,)
        frame_i_indices = frame_pairs_dict['frame_i_indices'].to(device=device, dtype=torch.long)

        # 4. Batched pose-encoding compute. We still loop over the L layers,
        # but that is far cheaper than per-pair (N_pairs * L) work.

        for layer_idx in range(num_layers):
            # (N_pairs, 16) @ (16, D) = (N_pairs, D)
            pose_encs_for_layer = T_rels_flat_batch @ self.camera_pose_embedding[layer_idx]

            # 5. Batched scatter (atomic add) into frame_i positions.
            # (S, D).index_add_(dim=0, index=(N_pairs,), tensor=(N_pairs, D))
            pose_encodings[layer_idx].index_add_(
                0,
                frame_i_indices,
                pose_encs_for_layer
            )

        # 6. Restore the (B, S, D) shape.
        # (L, S, D) -> List[(1, S, D)]
        final_pose_encodings = [pe.unsqueeze(0) for pe in pose_encodings]

        return final_pose_encodings
    
    

    def forward(
        self,
        multilevel_features: List[torch.Tensor],
        patch_start_idx: int = 0,
        frame_pairs: Optional[List[List[Dict]]] = None,  # 🆕
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for VGGT predictions (self depth prediction).

        Args:
            multilevel_features: List of (B, S, num_tokens, vision_hidden_size)
            patch_start_idx: Starting index for patches
            frame_pairs: Optional frame pairs (not used for self depth prediction,
                kept for API compatibility)

        Note:
            This method predicts each frame's OWN depth using its own features,
            WITHOUT pose encoding. For cross-view depth prediction (using frame_i
            features to predict frame_j depth), use forward_cross_view_depth().
        """
        if not multilevel_features or len(multilevel_features) == 0:
            raise ValueError("multilevel_features cannot be empty")

        # multilevel_features actually has shape (B, S, num_tokens, hidden);
        # infer B and S from the first feature tensor.
        first_feat_shape = multilevel_features[0].shape
        B = first_feat_shape[0]  # Batch size
        S = first_feat_shape[1]  # Number of frames
        H, W = self.resolution, self.resolution

        predictions = {}

        # For self depth prediction, do NOT apply pose encoding
        # Pose encoding is only used for cross-view depth prediction
        aggregated_tokens_list = self.prepare_vggt_features(
            multilevel_features, frame_pairs, B=B, S=S, apply_pose_encoding=False
        )

        if self.enable_depth and hasattr(self, 'depth_head'):
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                patch_start_idx=patch_start_idx,
                B=B, S=S, H=H, W=W
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf

        if self.enable_points and hasattr(self, 'point_head'):
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list,
                patch_start_idx=patch_start_idx,
                B=B, S=S, H=H, W=W
            )
            predictions["world_points"] = pts3d
            predictions["world_points_conf"] = pts3d_conf

            # Debug: print DPT output statistics on first forward pass (rank 0 only)
            if not hasattr(self, '_dpt_output_stats_printed'):
                if safe_get_rank() == 0:
                    print(f"  world_points:      mean={pts3d.abs().mean().item():.6f}, max={pts3d.abs().max().item():.4f}")
                    print(f"  world_points_conf: mean={pts3d_conf.abs().mean().item():.6f}, max={pts3d_conf.abs().max().item():.4f}")
                self._dpt_output_stats_printed = True

        # FIX: Immediately clean up large intermediate tensors to reduce memory fragmentation
        # aggregated_tokens_list is ~380MB (4 layers × B × S × H × W × 2048)
        # Keeping it in memory causes fragmentation over time, leading to training slowdown
        # The computation graph is already saved in depth/pts3d outputs, so gradient flow is unaffected
        del aggregated_tokens_list

        return predictions

    def forward_cross_view_depth(
        self,
        multilevel_features: List[torch.Tensor],
        frame_pairs: Dict[str, torch.Tensor],
        patch_start_idx: int = 0,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Cross-view depth prediction: use frame_i features + pose_enc(i→j) to predict frame_j depth.

        This method implements the cross-view depth prediction where:
        - For each pair (i, j), we take frame_i's features
        - Add pose encoding for the transformation i→j
        - Predict the depth at frame_j's viewpoint

        Pairs with zero overlap (valid_mask all zeros) are skipped to save compute.
        The returned 'valid_pair_indices' maps output predictions back to the original
        frame_pairs indices so the loss function can align correctly.

        Args:
            multilevel_features: List of (B, S, num_tokens, vision_hidden_size) tensors
            frame_pairs: Dict containing:
                - 'num_pairs': int, total number of pairs
                - 'frame_i_indices': (N_pairs,) tensor of source frame indices
                - 'frame_j_indices': (N_pairs,) tensor of target frame indices
                - 'T_relatives': (N_pairs, 4, 4) tensor of relative transformations
                - 'batch_indices': (N_pairs,) tensor of batch indices
                - 'valid_masks': (N_pairs, H, W) tensor of overlap masks (optional)
            patch_start_idx: Starting index for patches

        Returns:
            Dict containing:
                - 'cross_view_depth': (N_valid, H, W, 1) predicted depth at frame_j viewpoint
                - 'cross_view_depth_conf': (N_valid, H, W) confidence for predictions
                - 'valid_pair_indices': (N_valid,) indices into the original frame_pairs
            Returns None if no valid pairs or depth_head not enabled.
        """
        if not self.enable_depth or not hasattr(self, 'depth_head'):
            return None

        if frame_pairs is None or frame_pairs.get('num_pairs', 0) == 0:
            return None

        if not multilevel_features or len(multilevel_features) == 0:
            raise ValueError("multilevel_features cannot be empty")

        # Get dimensions
        first_feat_shape = multilevel_features[0].shape
        B = first_feat_shape[0]
        S = first_feat_shape[1]
        num_tokens = first_feat_shape[2]
        H, W = self.resolution, self.resolution

        num_pairs = frame_pairs['num_pairs']
        device = multilevel_features[0].device
        dtype = multilevel_features[0].dtype

        # --- Early filtering: skip pairs with no overlap ---
        valid_masks = frame_pairs.get('valid_masks', None)
        if valid_masks is not None:
            has_overlap = valid_masks.view(num_pairs, -1).any(dim=1)  # (N_pairs,)
            valid_pair_indices = torch.where(has_overlap)[0].to(device=device)
            num_valid = valid_pair_indices.shape[0]

            if num_valid == 0:
                return None

        else:
            # No valid_masks available, keep all pairs
            valid_pair_indices = torch.arange(num_pairs, device=device)
            num_valid = num_pairs

        # Get indices (filtered to valid pairs only)
        frame_i_indices = frame_pairs['frame_i_indices'].to(device=device, dtype=torch.long)[valid_pair_indices]
        batch_indices = frame_pairs['batch_indices'].to(device=device, dtype=torch.long)[valid_pair_indices]

        # Get T_relatives and compute pose encodings for each pair
        T_rels = frame_pairs['T_relatives'].to(device=device, dtype=dtype)
        T_rels_flat = T_rels[valid_pair_indices].view(num_valid, -1)

        # Compute pose encodings for each pair: (N_valid, 16) @ (16, D) = (N_valid, D)
        num_layers = len(self.intermediate_layer_idx)
        pose_encodings_per_pair = []
        for layer_idx in range(num_layers):
            pose_enc = T_rels_flat @ self.camera_pose_embedding[layer_idx].to(device=device, dtype=dtype)
            pose_encodings_per_pair.append(pose_enc)

        # Extract frame_i features for each pair and add pose encoding
        max_layer_idx = max(self.intermediate_layer_idx)
        aggregated_tokens_list = [None] * (max_layer_idx + 1)

        for i, layer_idx in enumerate(self.intermediate_layer_idx):
            features = multilevel_features[i]  # (B, S, num_tokens, hidden)

            # Gather features for frame_i of each pair: (N_valid, num_tokens, hidden)
            if B == 1:
                features_i = features[0, frame_i_indices]
            else:
                features_i = features[batch_indices, frame_i_indices]

            # Add pose encoding: (N_valid, D) -> (N_valid, 1, D) for broadcasting
            features_i = features_i + pose_encodings_per_pair[i].unsqueeze(1)

            # Project through feature_projector
            projected = self.feature_projector(features_i)  # (N_valid, num_tokens, 2048)
            aggregated_tokens_list[layer_idx] = projected.unsqueeze(0)  # (1, N_valid, num_tokens, 2048)

        # Run through depth_head
        # depth_head expects (B, S, ...) format, so we treat N_valid as S
        cross_depth, cross_depth_conf = self.depth_head(
            aggregated_tokens_list,
            patch_start_idx=patch_start_idx,
            B=1, S=num_valid, H=H, W=W
        )

        # cross_depth shape: (1, N_valid, H, W, 1) -> (N_valid, H, W, 1)
        # cross_depth_conf shape: (1, N_valid, H, W) -> (N_valid, H, W)
        cross_depth = cross_depth.squeeze(0)
        cross_depth_conf = cross_depth_conf.squeeze(0)

        # Clean up
        del aggregated_tokens_list

        return {
            'cross_view_depth': cross_depth,
            'cross_view_depth_conf': cross_depth_conf,
            'valid_pair_indices': valid_pair_indices,
        }
