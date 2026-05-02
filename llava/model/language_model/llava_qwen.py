#    Copyright 2024 Hao Zhang
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


from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, extract_visual_tokens
from transformers import Qwen2Config
from .qwen2.modeling_qwen2 import Qwen2Model, Qwen2ForCausalLM

import torch.distributed as dist
from deepspeed.comm import get_rank
from llava.model.geometric_loss import MultitaskLoss, check_and_fix_inf_nan

def gather_loss(loss):
    # wrap the scalar loss into a tensor on the current device
    loss_tensor = torch.tensor([loss], device=torch.cuda.current_device())

    # gather loss tensors from all ranks
    world_size = dist.get_world_size()
    gathered_losses = [torch.zeros_like(loss_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_losses, loss_tensor)

    # convert back to a list of Python scalars
    gathered_losses = [loss.item() for loss in gathered_losses]
    return gathered_losses

class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if hasattr(config, "ground_head_type") and config.ground_head_type is not None:
            self.ground_head_type = config.ground_head_type
            if config.ground_head_type == "mlp":
                self.ground_head = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size)
                )
            elif config.ground_head_type == "score":
                self.ground_head_temperature = config.ground_head_temperature
                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                )
                self.ground_head_score = nn.Sequential(
                    nn.Linear(1024, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1),
                )
            elif config.ground_head_type == "infonce":
                # self.ground_head_temperature = nn.Parameter(torch.tensor(config.ground_head_temperature))
                try:
                    self.ground_head_temperature = config.ground_head_temperature
                except:
                    self.ground_head_temperature = 0.07
                self.ground_head_zero_target = torch.nn.Parameter(torch.randn(config.hidden_size))
                self.empty_object_embedding = torch.nn.Parameter(torch.randn(config.hidden_size) * 0.02)
                self.ground_head_obj = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
                self.ground_head_query = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size),
                    nn.Linear(config.hidden_size, config.hidden_size),
                )
            else:
                raise NotImplementedError
        # Store the loss module instance directly to ensure a static model graph
        # Note: coord loss is now handled directly (like distill_loss), not through MultitaskLoss
        self.geometric_loss_module = MultitaskLoss(
            camera=None,
            depth={'weight': 0.8, 'gradient_loss_fn': 'grad', 'valid_range': 0.98},
            point={'weight': 0.1, 'gradient_loss_fn': 'normal', 'valid_range': 0.98},
            track=None,
            use_cross_view=True  # 🆕 Enable cross-view depth consistency loss
        ) if getattr(config, 'use_depth_auxiliary_task', False) else None

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        video_dict=None,
        use_object_proposals: bool = False,
        box_labels = None,
        img_pos_list = None,
        img_length_list = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, object_features, object_boxes, img_pos_list, img_length_list) = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    modalities,
                    image_sizes,
                    video_dict,
                    use_object_proposals=use_object_proposals,
                )
        if use_object_proposals:
            return self.predict_box(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                object_features=object_features,
                object_boxes=object_boxes,
                box_labels=box_labels,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict,
            )
            

        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                img_pos_list=img_pos_list,
                img_length_list=img_length_list,
                video_dict=video_dict
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _, _, _, img_pos_list, img_length_list) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes, video_dict=kwargs.get("video_dict", None))
            kwargs['img_pos_list'] = img_pos_list
            kwargs['img_length_list'] = img_length_list
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        img_pos_list = kwargs.pop("img_pos_list", None)
        img_length_list = kwargs.pop("img_length_list", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if img_pos_list is not None:
            inputs["img_pos_list"] = img_pos_list
        if img_length_list is not None:
            inputs["img_length_list"] = img_length_list

        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    def geometric_loss(
        self,
        prediction,
        batch_gt,
    ):
        # Use the pre-instantiated loss module.
        if self.geometric_loss_module is None:
            return None

        geometric_loss_dict = self.geometric_loss_module(prediction, batch_gt)
        return geometric_loss_dict

    def feature_3d_alignment(self, video_dict, hidden_states, img_pos_list=None, img_length_list=None, margin=0.0):
        # If feature_3d is not in video_dict, return 0 (no distillation loss)
        if 'feature_3d' not in video_dict or video_dict.get('feature_3d') is None:
            return 0.0

        C = hidden_states.shape[-1]
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🚀 OPTIMIZATION: Reduce memory fragmentation by minimizing .contiguous() calls
        # Use reshape() which handles non-contiguous tensors automatically
        # and creates fewer intermediate copies
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        # Extract and reshape in one step using slicing + reshape (avoids intermediate contiguous)
        feature_slice = hidden_states[:,img_pos_list[0]:img_pos_list[0]+img_length_list[0],:]
        # Use narrow + reshape pattern to minimize copies
        feature = feature_slice.view(32, 14, 15, C)[:, :, :14, :].reshape(32*14*14, C)

        feature_proj = self.model.proj_3d(feature)
        feature_3d = video_dict['feature_3d']
        feature_3d = feature_3d.to(device=feature.device, dtype=feature.dtype)
        feature_3d = feature_3d.squeeze()

        S, L, D = feature_3d.shape
        assert feature_proj.shape[-1] == D and S == 32
        if L == 768:
            feature_3d = feature_3d.view(S, 24, 32, D).permute(0, 3, 1, 2)
        elif L == 1036:
            feature_3d = feature_3d.view(S, 28, 37, D).permute(0, 3, 1, 2)
        elif L == 256:
            feature_3d = feature_3d.view(S, 16, 16, D).permute(0, 3, 1, 2)
        else:
            raise NotImplementedError

        # Pool and reshape in fewer steps - pool output is contiguous
        feature_3d = F.adaptive_avg_pool2d(feature_3d.contiguous(), (14, 14))  # Single contiguous call
        feature_3d = feature_3d.permute(0, 2, 3, 1).reshape(S*14*14, D)  # reshape handles layout
        feature_proj_norm = F.normalize(feature_proj, p=2, dim=-1)
        feature_3d_norm = F.normalize(feature_3d, p=2, dim=-1)

        # L2 distance computation
        feature_sim = ((feature_proj_norm - feature_3d_norm.detach()) ** 2).sum(dim=-1)
        feature_sim_loss = feature_sim.mean()
        
        return feature_sim_loss
    
    def predict_box(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        video_dict=None,
        object_features=None,
        object_boxes=None,
        box_labels=None,
        img_pos_list=None,
        img_length_list=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            img_pos_list=img_pos_list,
            img_length_list=img_length_list
        )

        hidden_states = outputs[0]
        ground_locations = (labels >= self.config.ground_token_ids[0]) & (labels <= self.config.ground_token_ids[-1])
        ground_hidden = hidden_states[ground_locations].squeeze(1)

        # ========== Extract visual tokens from LLM hidden states ==========
        S = img_length_list[0] // (14 * 15)  # num_frames
        C = hidden_states.shape[-1]
        visual_tokens_llm = extract_visual_tokens(
            hidden_states=hidden_states, start_pos=img_pos_list[0], length=img_length_list[0], num_frames=S, flatten=True
        )  # [S, 196, C]
        visual_tokens_llm = visual_tokens_llm.view(-1, C)  # [V, C]

        if self.ground_head_type == "infonce":
            object_features = torch.cat([object_features, self.ground_head_zero_target.unsqueeze(0)], dim=0)
            obj_feat = self.ground_head_obj(object_features.to(ground_hidden.dtype))
            query_feat = self.ground_head_query(ground_hidden)
            obj_feat = F.normalize(obj_feat)
            query_feat = F.normalize(query_feat)
            scores = (obj_feat * query_feat).sum(dim=-1)

        loss = None
        if box_labels is not None:
            if self.ground_head_type == "infonce":
                if len(box_labels[0]) == 0: # zero-target
                    box_labels[0].append(-1)
                logits = torch.exp(scores / self.ground_head_temperature)
                loss = - torch.log( logits[box_labels[0]].sum() / logits.sum())

            if loss is not None:
                _mem_diag = {}
                _mem_diag['before_distill'] = torch.cuda.memory_allocated() / 1024**3

                # Track video_dict data characteristics
                if video_dict is not None:
                    # Number of frames
                    if 'depth_maps' in video_dict and video_dict['depth_maps'] is not None:
                        dm = video_dict['depth_maps']
                        _mem_diag['depth_maps_shape'] = list(dm.shape) if hasattr(dm, 'shape') else 'N/A'
                    if 'depth_coords' in video_dict and video_dict['depth_coords'] is not None:
                        dc = video_dict['depth_coords']
                        _mem_diag['depth_coords_shape'] = list(dc.shape) if hasattr(dc, 'shape') else 'N/A'
                    if 'world_coords' in video_dict and video_dict['world_coords'] is not None:
                        wc = video_dict['world_coords']
                        if isinstance(wc, (list, tuple)) and len(wc) > 0:
                            _mem_diag['world_coords_shape'] = list(wc[0].shape) if hasattr(wc[0], 'shape') else 'N/A'
                    if 'frame_pairs' in video_dict and video_dict['frame_pairs'] is not None:
                        fp = video_dict['frame_pairs']
                        if hasattr(fp, 'shape'):
                            _mem_diag['frame_pairs_shape'] = list(fp.shape)
                        elif isinstance(fp, (list, tuple)):
                            _mem_diag['frame_pairs_count'] = len(fp)
                    if 'feature_3d' in video_dict and video_dict['feature_3d'] is not None:
                        f3d = video_dict['feature_3d']
                        _mem_diag['feature_3d_shape'] = list(f3d.shape) if hasattr(f3d, 'shape') else 'N/A'
                    if 'dpt_predictions' in video_dict and video_dict['dpt_predictions'] is not None:
                        dpt = video_dict['dpt_predictions']
                        if isinstance(dpt, dict):
                            _mem_diag['dpt_keys'] = list(dpt.keys())
                            for k, v in dpt.items():
                                if hasattr(v, 'shape'):
                                    _mem_diag[f'dpt_{k}_shape'] = list(v.shape)

                # Feature 3D distillation loss (only when feature_3d is present)
                distill_loss = None
                if 'feature_3d' in video_dict and video_dict['feature_3d'] is not None:
                    distill_loss = self.feature_3d_alignment(video_dict, hidden_states, img_pos_list=img_pos_list, img_length_list=img_length_list)
                    loss += distill_loss
                _mem_diag['after_distill'] = torch.cuda.memory_allocated() / 1024**3

                # Only compute geometric loss if enabled
                total_depth_loss = None
                if hasattr(self, 'geometric_loss') and self.geometric_loss_module is not None:
                    _mem_diag['before_geometric'] = torch.cuda.memory_allocated() / 1024**3

                    gt_batch = {
                        "depth": video_dict['depth_maps'],
                        "world_points": video_dict['depth_coords']
                    }
                    # 🆕 Add frame_pairs for cross-view consistency loss
                    if 'frame_pairs' in video_dict:
                        gt_batch['frame_pairs'] = video_dict['frame_pairs']
                    # 🆕 Add cross_view_predictions for cross-view prediction loss
                    if 'cross_view_predictions' in video_dict:
                        gt_batch['cross_view_predictions'] = video_dict['cross_view_predictions']

                    # Track DPT predictions size and compute geometric loss
                    dpt_preds = video_dict.get('dpt_predictions', None)
                    if dpt_preds is not None:
                        if isinstance(dpt_preds, dict):
                            _mem_diag['dpt_pred_keys'] = list(dpt_preds.keys())
                        elif isinstance(dpt_preds, torch.Tensor):
                            _mem_diag['dpt_pred_shape'] = list(dpt_preds.shape)

                        geometric_loss_dict = self.geometric_loss(dpt_preds, gt_batch)
                        _mem_diag['after_geometric_loss'] = torch.cuda.memory_allocated() / 1024**3

                        if geometric_loss_dict is not None:
                            total_depth_loss = geometric_loss_dict.get("objective", torch.zeros((), device=loss.device, dtype=loss.dtype))
                            loss += total_depth_loss

                _mem_diag['final'] = torch.cuda.memory_allocated() / 1024**3

                # Store for slow step detection
                if not hasattr(self, '_mem_diagnostic'):
                    self._mem_diagnostic = {}
                self._mem_diagnostic = _mem_diag

                # Initialize counter if not exists
                if not hasattr(self, '_loss_print_counter'):
                    self._loss_print_counter = 0
                self._loss_print_counter += 1

                # Only gather and print every 10 steps
                PRINT_INTERVAL = 10
                if self._loss_print_counter % PRINT_INTERVAL == 0:
                    # ✅ Gather ALL losses BEFORE rank check (collective operation must be called by all ranks)
                    gathered_distill_loss = gather_loss(distill_loss) if distill_loss is not None else None

                    gathered_total_geom = None
                    gathered_depth_only = None
                    gathered_cross_view = None
                    gathered_cross_view_pred = None
                    gathered_point_only = None

                    if total_depth_loss is not None and geometric_loss_dict is not None:
                        gathered_total_geom = gather_loss(total_depth_loss)

                        if 'total_depth_loss' in geometric_loss_dict:
                            depth_loss_only = geometric_loss_dict['total_depth_loss']
                            gathered_depth_only = gather_loss(depth_loss_only)
                            loss_cross_view = geometric_loss_dict.get("loss_cross_view_depth", torch.zeros_like(depth_loss_only))
                            gathered_cross_view = gather_loss(loss_cross_view)

                            loss_cross_view_pred = geometric_loss_dict.get("loss_cross_view_pred", torch.zeros_like(depth_loss_only))
                            gathered_cross_view_pred = gather_loss(loss_cross_view_pred)

                        if 'total_point_loss' in geometric_loss_dict:
                            point_loss_only = geometric_loss_dict['total_point_loss']
                            gathered_point_only = gather_loss(point_loss_only)

                    # Now only rank 0 prints (after all ranks have gathered)
                    if get_rank() == 0:

                        # Build unified loss string (with "Grounding" prefix to distinguish from modeling_qwen2.py)
                        loss_parts = []
                        if gathered_distill_loss is not None:
                            avg_distill_loss = sum(gathered_distill_loss) / len(gathered_distill_loss)
                            loss_parts.append(f"Grounding Distill: {avg_distill_loss:.4f}")
                        if gathered_total_geom is not None:
                            avg_total_geom = sum(gathered_total_geom) / len(gathered_total_geom)
                            loss_parts.append(f"Grounding Geometric: {avg_total_geom:.4f}")

                        if loss_parts:
                            print(" | ".join(loss_parts))

                        # Print depth/point breakdown
                        if gathered_depth_only is not None:
                            avg_depth_only = sum(gathered_depth_only) / len(gathered_depth_only)
                            avg_cross_view = sum(gathered_cross_view) / len(gathered_cross_view)
                            avg_cross_view_pred = sum(gathered_cross_view_pred) / len(gathered_cross_view_pred)
                            print(f"  ├─ Depth Loss: {avg_depth_only:.4f} (cross-view: {avg_cross_view:.4f}, cross-view-pred: {avg_cross_view_pred:.4f})")

                        if gathered_point_only is not None:
                            avg_point_only = sum(gathered_point_only) / len(gathered_point_only)
                            print(f"  └─ Point Loss: {avg_point_only:.4f}")

        return loss, scores

AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
