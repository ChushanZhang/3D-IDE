import argparse
import copy
import torch
import torch.nn.functional as F

import os
import json
import ray
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import shortuuid
import fasteners

from transformers import AutoConfig
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, OBJECT_TOKEN, GROUND_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.video_utils import VideoProcessor, merge_video_dict

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX, OBJECT_TOKEN
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, has_object_token: bool = False, object_token_id: int = None, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    """
    Preprocess sources for Qwen model.

    Args:
        has_object_token: If True, add <object> token after <image> token for SCAN2CAP OBJECT_TOKEN feature
        object_token_id: The actual token ID for <object> from tokenizer (required if has_object_token=True)
    """
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
                    # 🆕 Add <object> token after <image> for SCAN2CAP
                    # Use real token ID from tokenizer (learned embedding, not placeholder)
                    if has_object_token and object_token_id is not None:
                        _input_id += [object_token_id]
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


@ray.remote(num_gpus=1)
def eval_model(scene_groups, args):
    """Evaluate model on a list of (video_id, questions) scene groups."""
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    config = {}
    if args.lora_path is not None:
        config = AutoConfig.from_pretrained(args.lora_path)
        config = config.to_dict()
    elif args.overwrite_cfg:
        config.update({
            'tie_word_embeddings': False,
            'use_cache': True,
            "vocab_size": 151648
        })

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, overwrite_config=config)

    if args.lora_path is not None:
        from transformers import AutoTokenizer
        from peft import PeftModel
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, args.lora_path, adapter_name="lora")
        model = model.merge_and_unload()
        state_dict = torch.load(os.path.join(args.lora_path, 'non_lora_trainables.bin'))
        msg = model.load_state_dict(state_dict, strict=False)

    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "a")
    file_lock = fasteners.InterProcessLock(ans_file)

    video_processor = VideoProcessor(
        video_folder=args.video_folder,
        annotation_dir=args.embodiedscan_folder,
        frame_sampling_strategy=args.frame_sampling_strategy,
    )
    # Eval does not use cross-view constraints; disable to save CPU time.
    video_processor.use_cross_view = False
    video_processor.cross_view_top_k = 0

    # Get <object> token ID from tokenizer
    object_token_ids = tokenizer.encode(OBJECT_TOKEN, add_special_tokens=False)
    if len(object_token_ids) == 1:
        object_token_id = object_token_ids[0]
        print(f"Found <object> token ID: {object_token_id}")
    else:
        object_token_id = None
        print(f"Warning: <object> token not found in tokenizer or has multiple IDs: {object_token_ids}")

    ret = []
    inference_time = []
    total_questions = sum(len(qs) for _, qs in scene_groups)
    pbar = tqdm(total=total_questions, desc="Inference")
    is_first_sample = True

    for video_id, scene_questions in scene_groups:
        # Load video data ONCE per scene (raw, before per-question fields)
        raw_video_dict = video_processor.process_3d_video(
            video_id,
            image_processor,
            force_sample=args.force_sample,
            frames_upbound=args.max_frame_num,
        )

        # Reuse video data for all questions in this scene
        for line in scene_questions:
            idx = line["id"]
            question_type = line["metadata"]["question_type"]
            dataset_name = line["metadata"]["dataset"]

            gt = line.get("annotations", [line["conversations"][1]["value"]])
            qs = line["conversations"][0]["value"]
            cur_prompt = args.extra_prompt + qs

            if line["box_input"] is not None:
                box_input = line["box_input"]

                if model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

                args.conv_mode = "qwen_1_5"

                conv = conv_templates[args.conv_mode].copy()

                # Insert <ground> token AFTER coordinates in the prompt text
                conversation_to_process = line["conversations"][0].copy()
                if box_input is not None:
                    coord_str = f"[{box_input[0]:.2f}, {box_input[1]:.2f}, {box_input[2]:.2f}]"
                    conversation_to_process["value"] = conversation_to_process["value"].replace(coord_str, coord_str + " " + GROUND_TOKEN)

                input_ids = preprocess_qwen([conversation_to_process,{'from': 'gpt','value': None}], tokenizer, has_image=True, has_object_token=False, object_token_id=None).cuda()

                # Calculate bbox token position for Cross Attention
                bbox_offset_from_object = None
                bbox_length = None
                if object_token_id is not None:
                    input_ids_flat = input_ids.squeeze(0)
                    right_bracket_id = tokenizer.encode(']', add_special_tokens=False)[0]

                    object_positions = (input_ids_flat == object_token_id).nonzero(as_tuple=True)[0]
                    if len(object_positions) > 0:
                        object_pos = object_positions[0].item()
                        prefix = input_ids_flat[:object_pos]
                        right_positions = (prefix == right_bracket_id).nonzero(as_tuple=True)[0]
                        if len(right_positions) > 0:
                            end_pos = right_positions[-1].item() + 1

                            start_pos = None
                            for search_start in range(end_pos - 1, max(0, end_pos - 25), -1):
                                token_text = tokenizer.decode([prefix[search_start].item()])
                                if '[' in token_text:
                                    start_pos = search_start
                                    break

                            if start_pos is not None:
                                bbox_offset_from_object = object_pos - start_pos
                                bbox_length = end_pos - start_pos

                # Debug: Print first sample's info
                if is_first_sample:
                    is_first_sample = False
                    print(f"[EVAL_DEBUG] box_input={box_input is not None}, object_token_id={object_token_id}")
                    print(f"[EVAL_DEBUG] modified prompt: {conversation_to_process['value'][:150]}...")
                    print(f"[EVAL_DEBUG] input_ids contains object_token: {object_token_id in input_ids.tolist()[0] if object_token_id else 'N/A'}")
                    print(f"[EVAL_DEBUG] bbox_offset_from_object={bbox_offset_from_object}, bbox_length={bbox_length}")

                # Create per-question copy with question-specific fields
                video_dict = copy.copy(raw_video_dict)
                video_dict["box_input"] = box_input
                video_dict["bbox_offset_from_object"] = bbox_offset_from_object
                video_dict["bbox_length"] = bbox_length

                # Compute bbox_patch_mask for debug (cosine similarity with GT)
                if box_input is not None and "world_coords" in video_dict:
                    world_coords = video_dict["world_coords"]  # [S, H, W, 3]
                    box_center = torch.tensor(box_input[:3], device=world_coords.device, dtype=world_coords.dtype)
                    box_size = torch.tensor(box_input[3:], device=world_coords.device, dtype=world_coords.dtype)
                    min_xyz = box_center - box_size / 2
                    max_xyz = box_center + box_size / 2
                    world_coords_new = world_coords[:, :378, :378, :].reshape(-1, 14, 27, 14, 27, 3).transpose(2, 3).flatten(3, 4)
                    target_patch = torch.all((min_xyz <= world_coords_new) & (world_coords_new <= max_xyz), dim=-1)
                    target_patch = target_patch.sum(dim=3) >= int(27 * 27 * 0.125)
                    video_dict["bbox_patch_mask"] = target_patch.view(-1)

                video_dict = merge_video_dict([video_dict])
                image_tensors = video_dict.pop('images').half().to(model.device)
                for k in video_dict:
                    if video_dict[k] is not None and hasattr(video_dict[k], 'half'):
                        video_dict[k] = video_dict[k].half().to(model.device)

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    start_time = time.time()
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensors,
                        modalities="video",
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_beams=args.num_beams,
                        max_new_tokens=512,
                        use_cache=True,
                        video_dict=video_dict,
                    )
                    inference_time.append(time.time() - start_time)

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[:-len(stop_str)]
                outputs = outputs.strip()
            else:
                # IoU < 0.5: skip inference, set empty prediction
                outputs = ""

            with file_lock:
                new_item = {
                        "dataset": dataset_name,
                        "sample_id": idx,
                        "prompt": cur_prompt,
                        "pred_response": outputs,
                        "gt_response": gt,
                        "model_id": model_name,
                        "question_type": question_type,
                        "scene": video_id
                }
                ret.append(new_item)
                ans_file.write(json.dumps(new_item) + "\n")
                ans_file.flush()

            pbar.update(1)

    pbar.close()
    ans_file.close()
    return inference_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video-folder", type=str, default="data")
    parser.add_argument("--embodiedscan-folder", type=str, default="data/embodiedscan")
    parser.add_argument("--frame_sampling_strategy", type=str, default="uniform")
    parser.add_argument("--extra-prompt", type=str, default="The video captures 3D spatial information of a scene. Please focus on the spatial relationships in the video and answer the following questions.\n")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--n_gpu", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    parser.add_argument("--max_frame_num", type=int, default=32)
    parser.add_argument("--force_sample", type=bool, default=True)
    parser.add_argument("--overwrite_cfg", type=bool, default=False)
    parser.add_argument("-n", type=int, default=-1)
    parser.add_argument("--lora-path", type=str, default=None)
    args = parser.parse_args()

    # Data
    with open(os.path.expanduser(args.question_file)) as f:
        questions = json.load(f)
    if args.n != -1:
        questions = questions[:args.n]

    if os.path.exists(args.answer_file):
        print(f"The {args.answer_file} already exists!!!")
        exit()

    # Group questions by scene to avoid redundant video loading
    scene_groups = defaultdict(list)
    for q in questions:
        scene_groups[q["video"]].append(q)
    scene_list = sorted(scene_groups.items(), key=lambda x: len(x[1]), reverse=True)

    print(f"Total: {len(questions)} questions across {len(scene_list)} scenes")

    ray.init()
    features = []
    for i in range(args.n_gpu):
        gpu_scenes = scene_list[i::args.n_gpu]
        features.append(eval_model.remote(gpu_scenes, args))

    ret = ray.get(features)
    inference_time = []
    for item in ret:
        inference_time.extend(item)

    print(f"time: {np.mean(inference_time)}")
