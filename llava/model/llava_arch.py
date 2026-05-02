#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
from llava.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP
import torch
import torch.nn as nn
import torch.nn.functional as F
from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random
from torch_scatter import scatter_mean
from llava.model.vggt_module import VGGTModule, NoOpDPTModule


def extract_visual_tokens(hidden_states=None, start_pos=None, length=None, num_frames=None, flatten=True,
                          world_coords=None, bboxes=None):
    """
    Extract visual tokens and/or compute bbox patch masks.

    The grid layout is (14, 15) per frame where the 15th column contains newline tokens.
    This function removes those newline tokens to get (14, 14) = 196 visual tokens per frame.

    Usage patterns:
        1. Extract visual tokens only:
           tokens = extract_visual_tokens(hidden_states, start_pos, length, num_frames)

        2. Compute bbox mask only:
           mask = extract_visual_tokens(world_coords=world_coords, bboxes=bboxes)

        3. Both:
           tokens, mask = extract_visual_tokens(hidden_states, start_pos, length, num_frames,
                                                 world_coords=world_coords, bboxes=bboxes)

    Args:
        hidden_states: Input tensor [seq_len, C] or [B, seq_len, C]
        start_pos: Start position of image tokens
        length: Length of image tokens (should be num_frames * 14 * 15)
        num_frames: Number of frames (required if hidden_states provided)
        flatten: If True, return (S, 196, C); if False, return (S, 14, 14, C)
        world_coords: [S, H, W, 3] where H, W >= 378
        bboxes: [6] for single bbox or [N, 6] for multiple bboxes (center_xyz, size_xyz)

    Returns:
        If only hidden_states: visual_tokens
        If only world_coords+bboxes: bbox_mask
        If both: (visual_tokens, bbox_mask)
    """
    visual_tokens = None
    bbox_mask = None

    # Extract visual tokens from hidden_states
    if hidden_states is not None:
        if hidden_states.dim() == 2:
            visual_raw = hidden_states[start_pos:start_pos+length]
        else:
            visual_raw = hidden_states[:, start_pos:start_pos+length, :].squeeze(0)

        C = visual_raw.shape[-1]
        visual_tokens = visual_raw.view(num_frames, 14, 15, C)[:, :, :14, :]

        if flatten:
            visual_tokens = visual_tokens.reshape(num_frames, 14*14, C)

    # Compute bbox patch mask from world_coords
    if world_coords is not None and bboxes is not None:
        # Debug: print shapes before reshape
        # print(f"[DEBUG extract_visual_tokens] world_coords.shape={world_coords.shape}, bboxes.shape={bboxes.shape}")
        # Reshape world_coords: [S, H, W, 3] -> [S, 14, 14, 729, 3]
        world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)
        threshold_count = int(27 * 27 * 0.125)  # 12.5% threshold

        if bboxes.dim() == 1:
            # Single bbox: [6] -> min_xyz [3], max_xyz [3]
            min_xyz = bboxes[:3] - bboxes[3:6] / 2
            max_xyz = bboxes[:3] + bboxes[3:6] / 2
            in_box = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)
            bbox_mask = in_box.sum(dim=3) >= threshold_count  # [S, 14, 14]
        else:
            # Multiple bboxes: [N, 6]
            N = bboxes.shape[0]
            centers = bboxes[:, :3]
            sizes = bboxes[:, 3:]
            min_xyz = centers - sizes / 2
            max_xyz = centers + sizes / 2

            min_xyz_exp = min_xyz.reshape(N, 1, 1, 1, 1, 3)
            max_xyz_exp = max_xyz.reshape(N, 1, 1, 1, 1, 3)
            world_coords_exp = world_coords_new.unsqueeze(0)

            in_box = torch.all((min_xyz_exp <= world_coords_exp) & (world_coords_exp <= max_xyz_exp), dim=-1)
            bbox_mask = in_box.sum(dim=-1) >= threshold_count  # [N, S, 14, 14]

    # Return based on what was computed
    if visual_tokens is not None and bbox_mask is not None:
        return visual_tokens, bbox_mask
    elif visual_tokens is not None:
        return visual_tokens
    else:
        return bbox_mask


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            # for name, param in self.vision_tower.named_parameters():
            #     if 'adapter_linear_up' in name:
            #         torch.nn.init.zeros_(param)
            #     elif 'adapter' in name:
            #         torch.nn.init.kaiming_normal_(param)
            #         print('initialized adapter weight', name, param)

            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        
            self.use_depth_auxiliary_task = getattr(config, 'use_depth_auxiliary_task', False)
            self.dpt_module = NoOpDPTModule()

            if getattr(self.config, 'world_position_embedding_type', None) is not None:
                if "sample9" in self.config.world_position_embedding_type:
                    n_points = 9
                elif "sample5" in self.config.world_position_embedding_type:
                    n_points = 5
                elif "minmax" in self.config.world_position_embedding_type:
                    n_points = 2
                else:
                    n_points = 1

                if "mlp" in self.config.world_position_embedding_type:
                    self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size, n_points=n_points)
                elif "sin3d" in self.config.world_position_embedding_type:
                    self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size, n_points=n_points)



    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        # Initialize DPT depth module for auxiliary training task
        # Note: Don't check self.training here - model is in eval mode during __init__
        if self.use_depth_auxiliary_task:
            weak_geometric_validator = getattr(self.config, 'weak_geometric_validator', False)
            use_pretrained = not weak_geometric_validator
            self.dpt_module = VGGTModule(
                vision_hidden_size=vision_tower.config.hidden_size,
                checkpoint_path=getattr(self.config, 'dpt_checkpoint_path', None),
                use_pretrained=use_pretrained,
                patch_size=vision_tower.config.patch_size,
                resolution=vision_tower.config.image_size,
                enable_depth=True,
                enable_points=False,
                llm_hidden_size=self.config.hidden_size,  # Pass LLM hidden size for CoordSupervisionModule
            )
            
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)

        return image_feature

    def average_coordinate_in_patch(self, world_coords, patch_size=27):
        V, H, W, D = world_coords.size()  # D = 3
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]
        world_coords = world_coords.permute(0, 3, 1, 2)
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)
        return world_coords_avg

    def minmax_coordinate_in_patch(self, world_coords, patch_size=27):
        V, H, W, D = world_coords.size()
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]
        world_coords = world_coords.permute(0, 3, 1, 2)
        world_coords_max = torch.nn.functional.max_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)
        world_coords_min = -torch.nn.functional.max_pool2d(-world_coords, kernel_size=patch_size, stride=patch_size)
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)
        return torch.stack([world_coords_min, world_coords_max], dim=3)

    def sample_n_points(self, world_coords, n_points=9):
        V, H, W, D = world_coords.size()
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]
        world_coords = world_coords.view(-1, 14, 27, 14, 27, 3).permute(0, 1, 3, 2, 4, 5)
        if n_points == 9:
            return world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
        elif n_points == 5:
            sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            return sample[:, :, :, 0::2, :].reshape(V, 14, 14, 5, 3)
        elif n_points == 1:
            sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            return sample[:, :, :, 4, :].reshape(V, 14, 14, 3)
        else:
            raise NotImplementedError

    def discrete_coords(self, world_coords, xyz_min, voxel_size=0.1):
        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.max_xyz_range).to(world_coords.device)
        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / voxel_size
        return world_coords_discrete.round().detach()

    def encode_images(self, images, world_coords=None):
        # layers.layer_norm1.weight
        multi_level_image_features = self.get_model().get_vision_tower()(images, world_coords)
        image_features = multi_level_image_features[-1]
        # print('vision weights:', self.get_model().get_vision_tower().vision_tower.vision_model.encoder.layers[0].layer_norm1.weight.grad)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        #print('image_features sizes', image_features.shape)
        return multi_level_image_features, image_features


    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat != 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature
    
    def ravel_hash_vec(self, arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert len(arr.shape) == 3
        arr -= arr.min(1, keepdims=True)[0]
        arr_max = arr.max(1, keepdims=True)[0] + 1

        keys = torch.zeros(arr.shape[0], arr.shape[1], dtype=arr.dtype).to(arr.device)

        # Fortran style indexing
        for j in range(arr.shape[2] - 1):
            keys += arr[..., j]
            keys *= arr_max[..., j + 1]
        keys += arr[..., -1]
        return keys
    
    def voxelization(self, xyz):
        """
        Inputs:
            xyz: tensor [B, N, 3]
        Outputs: 
            point_to_voxel_all: tensor [B, N], is the mapping from original point cloud to voxel
        """
        B, N, _ = xyz.shape
        # xyz = xyz / voxel_size
        # xyz = torch.round(xyz).long()
        # xyz = xyz - xyz.min(1, keepdim=True)[0]

        keys = self.ravel_hash_vec(xyz)

        # Optimize: avoid repeated dtype conversion
        keys_long = keys.long()
        point_to_voxel = torch.stack(
            [torch.unique(keys_long[b], return_inverse=True)[1] for b in range(B)], 0)
        return point_to_voxel

    def voxel_pooling(self, world_coords_discrete, image_feature):
        # world_coords_discrete: [V H W D]
        # image_feature: [V N C]
        world_coords_discrete = world_coords_discrete[0]
        image_feature = image_feature[0]

        V, H, W, D = world_coords_discrete.size()
        world_coords_discrete = world_coords_discrete.view(1, V*H*W, D)
        # print('voxel numbers:', (world_coords_discrete.max(dim=1)[0] - world_coords_discrete.min(dim=1)[0]))
        _, N, C = image_feature.size()
        assert H*W == N and image_feature.shape[0] == V and D == 3
        image_feature = image_feature.contiguous().view(1, V*H*W, C)
        p2v = self.voxelization(world_coords_discrete)
        pooled_video_features = torch.cat([scatter_mean(image_feature[b], p2v[b], dim=0) for b in range(len(image_feature))]) # bn, F
        # print('pooled_video_features shape: ', pooled_video_features.shape)
        # pooled_video_features = pooled_video_features.unsqueeze(0)

        return pooled_video_features
        
    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
        use_clip_refer=True
    ):
        if video_dict is not None and input_ids is not None:
            video_dict['original_input_ids'] = input_ids.clone()

        use_sin3d_pe = False
        use_mlp_pe = False
        world_coords_patch14_discrete = None
        skip_3d_pe_fusion = getattr(self.config, 'skip_3d_pe_fusion', True)
        if getattr(self.config, 'world_position_embedding_type', None) is not None and past_key_values is None and not skip_3d_pe_fusion:
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if 'box_input' in video_dict and len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords_patch14 = [self.average_coordinate_in_patch(coords, patch_size=14) for coords in world_coords]
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type:
                world_coords_discrete = [self.discrete_coords(coords, xyz_min[i], voxel_size=self.config.voxel_size) for i, coords in enumerate(world_coords)]
                if 'avg' in self.config.world_position_embedding_type:
                    world_coords_patch14_discrete = [self.discrete_coords(coords, xyz_min[i], voxel_size=self.config.voxel_size*2) for i, coords in enumerate(world_coords_patch14)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input, None, voxel_size=self.config.voxel_size)

            if "sin3d" in self.config.world_position_embedding_type:
                use_sin3d_pe = True

            if "mlp" in self.config.world_position_embedding_type:
                use_mlp_pe = True


        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = [i for i, m in enumerate(modalities) if m == "video"]

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            multi_level_feats, encoded_image_features = self.encode_images(concat_images, world_coords_patch14_discrete)

            # image_features,all_faster_video_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            # print(encoded_image_features.shape, split_sizes)
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            # Use list comprehension - more efficient than loop
            image_features = [
                self.get_2dPool(feat) if idx in video_idx_in_batch else feat
                for idx, feat in enumerate(encoded_image_features)
            ]
            object_boxes = None
            gt_bbox_masks_for_grounding = None  # GT masks for grounding (used in predict_box)
            world_coords = None  # Initialize

            if use_object_proposals and use_clip_refer:
                object_boxes = video_dict["objects"][0]  # [N, 6]
                obj_num = len(object_boxes)

                world_coords_raw = video_dict.get('world_coords', None)
                if world_coords_raw is not None:
                    world_coords = world_coords_raw[0] if isinstance(world_coords_raw, (list, tuple)) else world_coords_raw
                    # Handle 5D tensor [B, S, H, W, 3] -> [S, H, W, 3]
                    if isinstance(world_coords, torch.Tensor) and world_coords.dim() == 5:
                        world_coords = world_coords[0]
                    gt_bbox_masks_for_grounding = extract_visual_tokens(
                        world_coords=world_coords, bboxes=object_boxes
                    )  # [N, S, 14, 14]
                    gt_bbox_masks_for_grounding = gt_bbox_masks_for_grounding.view(obj_num, -1)  # [N, V]
                    # Store for reuse (avoid redundant computation)
                    video_dict['gt_masks'] = gt_bbox_masks_for_grounding  # [N, V]

                # visual_tokens: flatten image_features [S, 196, H] -> [V, H]
                visual_tokens = image_features[0].reshape(-1, image_features[0].shape[-1])  # [V, H]

                # Extract GT features used downstream as object features.
                # detach() prevents GT features from affecting backbone gradients.
                if gt_bbox_masks_for_grounding is not None:
                    gt_object_features = []
                    for n in range(obj_num):
                        mask_n = gt_bbox_masks_for_grounding[n]  # [V] bool
                        if mask_n.any():
                            gt_feat = visual_tokens[mask_n].mean(dim=0).detach()  # [H]
                        else:
                            gt_feat = visual_tokens.mean(dim=0).detach()  # fallback
                        gt_object_features.append(gt_feat)
                    video_dict['gt_object_features'] = torch.stack(gt_object_features)  # [N, H]

                # Build object_patch from GT masks (used to index image_features below).
                object_patch = []
                if gt_bbox_masks_for_grounding is not None:
                    for l in range(obj_num):
                        cur_object_patch = gt_bbox_masks_for_grounding[l].view(-1, 14, 14)  # [S, 14, 14]
                        object_patch.append(cur_object_patch)

            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "one_token")

            if use_object_proposals and use_clip_refer:
                object_features = []
                valid_obj_num = 0

                if len(object_patch) == 0:
                    raise ValueError("GT bbox masks unavailable; world_coords must be provided when use_object_proposals=True.")
                for l in range(obj_num):
                    if "patch14" in self.config.object_feature_type:
                        cur_object_features = image_features[0][object_patch[l].view(-1, 196)]
                    else:
                        raise NotImplementedError

                    if len(cur_object_features) == 0:
                        cur_object_features = self.empty_object_embedding
                    else:
                        if cur_object_features.dim() > 1:
                            cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1
                    object_features.append(cur_object_features)

                object_features = torch.stack(object_features)
            else:
                object_features = None

            # NOTE: baseline (Visual-AI/3DRS) also injects PE into object_features
            # via `object_features += world_position_embedding(object_boxes_center)`.
            # That path is intentionally not restored here -- only image-token PE is fused.
            if use_sin3d_pe or use_mlp_pe:
                new_image_features = []
                for idx, image_feat in enumerate(image_features):
                    if "discrete" in self.config.world_position_embedding_type:
                        coords = world_coords_discrete[idx].flatten(1, 2)
                    else:
                        coords = world_coords[idx].flatten(1, 2)

                    image_feat = image_feat + self.get_model().world_position_embedding(coords.detach())
                    new_image_features.append(image_feat)
                image_features = new_image_features

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]

            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if mm_newline_position == "grid":
                            # Grid-wise
                            image_feature = self.add_token_per_grid(image_feature)
                            if getattr(self.config, "add_faster_video", False):
                                faster_video_feature = self.add_token_per_grid(all_faster_video_features[image_idx])
                                # Add a token for each frame
                                concat_slow_fater_token = []
                                # import pdb; pdb.set_trace()
                                for _ in range(image_feature.shape[0]):
                                    if _ % self.config.faster_token_stride == 0:
                                        concat_slow_fater_token.append(torch.cat((image_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                    else:
                                        concat_slow_fater_token.append(torch.cat((faster_video_feature[_], self.model.faster_token[None].to(image_feature.device)), dim=0))
                                # import pdb; pdb.set_trace()
                                image_feature = torch.cat(concat_slow_fater_token)

                                # print("!!!!!!!!!!!!")
                        
                            new_image_features.append(image_feature)
                        elif mm_newline_position == "frame":
                            # Frame-wise
                            image_feature = self.add_token_per_frame(image_feature)

                            new_image_features.append(image_feature.flatten(0, 1))
                            
                        elif mm_newline_position == "one_token":
                            # one-token
                            image_feature = image_feature.flatten(0, 1)
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                            new_image_features.append(image_feature)      
                        elif mm_newline_position == "no_token":
                            new_image_features.append(image_feature.flatten(0, 1))
                        else:
                            raise ValueError(f"Unexpected mm_newline_position: {mm_newline_position}")
                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2]
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        new_image_features.append(image_feature)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            multi_level_feats, image_features = self.encode_images(images)

        if self.training:
            multi_level_feats = [feat.unsqueeze(0) for feat in multi_level_feats]
            frame_pairs = video_dict.get('frame_pairs', None) if video_dict is not None else None

            dpt_module = getattr(self.get_model(), 'dpt_module')
            dpt_result = dpt_module(multi_level_feats, frame_pairs=frame_pairs)
            video_dict['dpt_predictions'] = dpt_result

            # 🆕 Cross-view depth prediction: auto-enabled when frame_pairs exist (same condition as consistency loss)
            if frame_pairs is not None:
                cross_view_preds = dpt_module.forward_cross_view_depth(multi_level_feats, frame_pairs)
                if cross_view_preds is not None:
                    video_dict['cross_view_predictions'] = cross_view_preds

            del multi_level_feats

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        img_pos_list = []
        img_length_list = []
        # rank_print("Inserting Images embedding")
        # rank0_print('feat:', voxel_pool_features.shape, image_features[0].shape)
        # image_features[0] = torch.cat([voxel_pool_features, image_features[0]],dim=0)
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            img_pos = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            img_pos_list.append(img_pos)
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_pos_index = 0
            mrope_pos_index = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                cur_new_world_coords.append(
                    torch.arange(cur_pos_index, cur_pos_index + len(cur_input_embeds_no_im[i])).to(cur_input_embeds_no_im[i].device).unsqueeze(1).repeat(1, 3)
                )
                cur_pos_index += len(cur_input_embeds_no_im[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    img_length = cur_image_features.shape[0]
                    img_length_list.append(img_length)

                    cur_image_idx += 1
                    # rank0_print('cur_image_features size:', cur_image_features.shape)
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            if video_dict is not None:
                img_length_cur = img_length_list[-1] if len(img_length_list) > 0 else 0

                if img_length_cur > 0:
                    num_frames = img_length_cur // (14 * 15)
                    H = image_features[0].shape[-1]
                    visual_raw = image_features[0].view(num_frames, 14, 15, H)  # [S, 14, 15, H]
                    visual_tokens = visual_raw[:, :, :14, :].reshape(-1, H)  # [S*196, H]

                    box_input = video_dict.get('box_input', None)
                    is_scan2cap = (box_input is not None and
                                   isinstance(box_input, torch.Tensor) and
                                   box_input.numel() >= 6)

                    if is_scan2cap:
                        if box_input.dim() == 2:
                            box_input = box_input[0]
                        bbox_coords = box_input.unsqueeze(0).to(visual_tokens.dtype)  # [1, 6]

                        ground_pos_new = None
                        ground_token_ids = getattr(self.config, 'ground_token_ids', None)
                        if ground_token_ids and len(ground_token_ids) > 0:
                            ground_token_id = ground_token_ids[0]
                            ground_mask = (cur_input_ids == ground_token_id)
                            if ground_mask.any():
                                ground_pos_original = ground_mask.nonzero(as_tuple=True)[0][0].item()
                                image_pos_original = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][0].item()
                                ground_pos_new = image_pos_original + img_length_cur + (ground_pos_original - image_pos_original - 1)
                                seq_len = cur_new_input_embeds.shape[0]
                                if not (0 <= ground_pos_new < seq_len):
                                    ground_pos_new = None

                        if ground_pos_new is None:
                            raise ValueError(f"SCAN2CAP: <ground> token not found! ground_token_ids={ground_token_ids}")

                        # Compute GT masks and features
                        gt_feat = None
                        world_coords_raw = video_dict.get('world_coords', None)
                        if world_coords_raw is not None:
                            world_coords = world_coords_raw[0] if isinstance(world_coords_raw, (list, tuple)) else world_coords_raw
                            if isinstance(world_coords, torch.Tensor) and world_coords.dim() == 5:
                                world_coords = world_coords[0]
                            gt_masks = extract_visual_tokens(world_coords=world_coords, bboxes=bbox_coords)
                            if gt_masks is not None:
                                gt_masks = gt_masks.view(1, -1).to(device=visual_tokens.device)  # [1, V]
                                video_dict['gt_masks'] = gt_masks

                                # detach() prevents GT features from affecting backbone gradients.
                                mask_0 = gt_masks[0]  # [V] bool
                                if mask_0.any():
                                    gt_feat = visual_tokens[mask_0].mean(dim=0).detach()  # [H]
                                else:
                                    gt_feat = visual_tokens.mean(dim=0).detach()  # fallback
                                video_dict['gt_object_features'] = gt_feat.unsqueeze(0)  # [1, H]

                        if gt_feat is not None:
                            cur_new_input_embeds = cur_new_input_embeds.clone()
                            cur_new_input_embeds[ground_pos_new] = gt_feat

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes, img_pos_list, img_length_list

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
