import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np
import torch
import requests

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist
import torch.nn.functional as F

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

# Cache meshgrids per (H, W, device) to avoid recreating them on every call.
_MESHGRID_CACHE = {}

# @torch.no_grad() prevents the whole function from accumulating into the autograd graph.
@torch.no_grad()
def compute_batched_reprojection_mask(
    depths_src,   # (N, H, W)
    T_relatives,  # (N, 4, 4)
    intrinsics,   # (N, 4, 4)
    H,
    W
):
    """
    Batched reprojection-mask helper.
    Computes (N) reprojection masks at once and uses bincount-style splatting.

    Args:
        depths_src: (N, H, W) source-frame depth maps in meters.
        T_relatives: (N, 4, 4) relative transforms T_{cam_i -> cam_j}
            mapping source-frame points into the target frame.
        intrinsics: (N, 4, 4) camera intrinsics.
        H, W: image size.

    Returns:
        final_valid_mask: (N, H, W) bool — which target pixels received a reprojection.
    """
    device = depths_src.device
    N = depths_src.shape[0]  # batch size

    # NOTE: callers (video_utils.py:301-306) already ensure inputs are float32,
    # so we skip a redundant dtype conversion here.

    # Extract intrinsics, reshaped to (N, 1, 1) for broadcasting.
    fx = intrinsics[:, 0, 0].view(N, 1, 1)
    fy = intrinsics[:, 1, 1].view(N, 1, 1)
    cx = intrinsics[:, 0, 2].view(N, 1, 1)
    cy = intrinsics[:, 1, 2].view(N, 1, 1)

    # Use a cached meshgrid to avoid recreating it each call.
    cache_key = (H, W, device)
    if cache_key not in _MESHGRID_CACHE:
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        # detach() so the global cache does not accumulate gradients.
        _MESHGRID_CACHE[cache_key] = (y.unsqueeze(0).detach(), x.unsqueeze(0).detach())

    y, x = _MESHGRID_CACHE[cache_key]

    # Step 1: source pixel + depth -> 3D in camera_src; all ops are (N, H, W).
    z_src = depths_src
    x_cam_src = (x - cx) * z_src / fx
    y_cam_src = (y - cy) * z_src / fy
    ones = torch.ones_like(x_cam_src)

    # Homogeneous coordinates: (N, H, W, 4)
    cam_coords_src = torch.stack([x_cam_src, y_cam_src, z_src, ones], dim=-1)

    # Step 2: transform into the target camera frame.
    # (N, H, W, 4) @ (N, 4, 4) cannot broadcast directly,
    # so flatten to (N, H*W, 4) @ (N, 4, 4) -> (N, H*W, 4).

    # (N, H*W, 4)
    cam_coords_src_flat = cam_coords_src.view(N, -1, 4)

    # (N, 4, 4)
    T_rel_T = T_relatives.transpose(1, 2)

    # Batched matmul (N, H*W, 4) @ (N, 4, 4) = (N, H*W, 4).
    cam_coords_tgt_flat = torch.matmul(cam_coords_src_flat, T_rel_T)

    # Restore (N, H, W, 4).
    cam_coords_tgt = cam_coords_tgt_flat.view(N, H, W, 4)

    x_tgt = cam_coords_tgt[..., 0]
    y_tgt = cam_coords_tgt[..., 1]
    z_tgt = cam_coords_tgt[..., 2]

    # Step 3: project to the target image plane (N, H, W).
    u_tgt = (x_tgt * fx / (z_tgt + 1e-6)) + cx
    v_tgt = (y_tgt * fy / (z_tgt + 1e-6)) + cy

    # Step 4: validity mask (N, H, W).
    valid_mask = (
        (u_tgt >= 0) & (u_tgt < W) &
        (v_tgt >= 0) & (v_tgt < H) &
        (z_tgt > 0.01) &
        (z_src > 0.01)
    )

    # Step 5: build the warped mask via batched bincount.
    # (N, H, W) -> (K,) where K is the total number of valid points across the batch.
    valid_u = u_tgt[valid_mask].long()
    valid_v = v_tgt[valid_mask].long()

    # Prefer nonzero() over expand() to avoid materializing a huge tensor.
    # The old approach allocated batch_idx.expand(N, H, W) (~4.7M elements);
    # nonzero() returns just the batch index of each valid point.
    valid_indices = valid_mask.nonzero(as_tuple=False)  # (K, 3) where K = num valid points
    batch_idx_flat = valid_indices[:, 0]  # (K,) — first column is the batch index

    # Use advanced indexing to avoid bincount(minlength=N*H*W)'s big allocation.
    # The old approach always allocated N*H*W (~4.7M) elements;
    # the new one only writes to the points we actually need.
    final_valid_mask = torch.zeros(N, H, W, dtype=torch.bool, device=device)

    # batch_idx_flat, valid_v, valid_u are the indices of the projected valid points;
    # mark those positions as True.
    final_valid_mask[batch_idx_flat, valid_v, valid_u] = True

    return final_valid_mask


def compute_cross_view_frame_pairs(
    poses,
    world_coords=None,
    valid_masks=None,
    mode='neighbor',
    top_k=1,
    min_baseline=0.1,
    max_baseline=2.0,
    distance_threshold=0.05,
    neighbor_range=1,  # how many neighbor frames to use
    neighbor_direction='forward',  # 'forward' / 'backward' / 'bidirectional'
):
    """
    Unified helper for computing cross-view frame pairs.

    Modes:
    - 'neighbor': use adjacent frames only (fastest, fallback).
    - 'overlap': pick frames by 3D overlap (slower, best quality).

    Args:
        poses: (V, 4, 4) camera poses (already aligned via axis_align_matrix).
        world_coords: (V, H, W, 3) GT world coords (required when mode='overlap').
        valid_masks: (V, H, W) valid pixel masks.
        mode: str, the selection mode.
        top_k: int, top-K best matches per frame (used for mode='heuristic'/'overlap').
        min_baseline: float, min camera translation (meters).
        max_baseline: float, max camera translation (meters).
        distance_threshold: float, distance threshold (meters) for overlap.
        neighbor_range: int, how many neighbor frames to use (default 1).
            - neighbor_range=1: frame_i -> frame_{i+1}
            - neighbor_range=2: frame_i -> [frame_{i+1}, frame_{i+2}]
        neighbor_direction: str, direction of pairing (only for mode='neighbor').
            - 'forward': use later frames only.
            - 'backward': use earlier frames only.
            - 'bidirectional': use both sides.

    Returns:
        frame_pairs: List of dict; each dict contains:
            - frame_i: int (source frame index)
            - frame_j: int (target frame index)
            - T_relative: (4, 4) tensor, T_{cam_i -> cam_j}
                computed as inverse(poses[j]) @ poses[i],
                mapping points from camera_i frame into camera_j frame.
            - type: str, 'neighbor' or 'best_match'
            - [optional] overlap: float
            - [optional] baseline: float
            - [optional] angle: float
    """
    V = poses.shape[0]

    # NOTE: callers in video_utils.py already cast poses/world_coords to float32,
    # so we skip a redundant dtype conversion here.

    # detach poses_inv so the autograd graph does not accumulate.
    poses_inv = torch.inverse(poses).detach()
    frame_pairs = []

    # ========== Mode 1: adjacent frames only (multi-target supported) ==========
    if mode == 'neighbor':
        # Multi-target neighbor pairing: each frame_i can pair with multiple frame_j's.
        for i in range(V):
            # Collect target-frame indices to pair against.
            target_frames = []

            if neighbor_direction in ['forward', 'bidirectional']:
                # Look forward: i+1, i+2, ..., i+neighbor_range.
                for offset in range(1, neighbor_range + 1):
                    j = i + offset
                    if j < V:  # boundary check
                        target_frames.append(j)

            if neighbor_direction in ['backward', 'bidirectional']:
                # Look backward: i-1, i-2, ..., i-neighbor_range.
                for offset in range(1, neighbor_range + 1):
                    j = i - offset
                    if j >= 0:  # boundary check
                        target_frames.append(j)

            # If there is no valid neighbor (e.g. last frame with direction='forward'),
            # fall back to identity transformation.
            if len(target_frames) == 0:
                T_identity = torch.eye(4, device=poses.device, dtype=poses.dtype).detach()
                frame_pairs.append({
                    "frame_i": i,
                    "frame_j": i,  # predict self
                    "T_relative": T_identity,
                    "type": "identity",
                })
            else:
                # Emit one pair for each target frame.
                for j in target_frames:
                    T_relative = torch.matmul(poses_inv[j], poses[i]).detach()
                    frame_pairs.append({
                        "frame_i": i,
                        "frame_j": j,
                        "T_relative": T_relative,
                        "type": "neighbor",
                        "offset": j - i,  # offset for debugging
                    })

        return frame_pairs

    # ========== Mode 2: precise selection by 3D overlap ==========
    elif mode == 'overlap':
        if world_coords is None or valid_masks is None:
            raise ValueError("mode='overlap' requires world_coords and valid_masks")

        # Track which frames already received pairs.
        frames_with_pairs = set()

        for i in range(V):
            candidates = []

            for j in range(V):
                if i == j:
                    continue

                # Compute baseline.
                t_i = poses[i][:3, 3]
                t_j = poses[j][:3, 3]
                baseline = torch.norm(t_i - t_j).item()

                if baseline < min_baseline or baseline > max_baseline:
                    continue

                # Compute overlap.
                overlap = _compute_frame_overlap(
                    world_coords[i], world_coords[j],
                    valid_masks[i], valid_masks[j],
                    distance_threshold
                )

                if overlap < 0.1:  # filter out near-zero overlap pairs
                    continue

                # Combined score.
                score = overlap * (1.0 - abs(baseline - 0.5) / max_baseline)

                # detach T_relative to keep it out of the autograd graph.
                T_relative = torch.matmul(poses_inv[j], poses[i]).detach()

                candidates.append({
                    "frame_j": j,
                    "T_relative": T_relative,
                    "overlap": overlap,
                    "baseline": baseline,
                    "score": score,
                })

            # Keep the top-K candidates.
            candidates.sort(key=lambda x: x["score"], reverse=True)
            selected_candidates = candidates[:top_k]

            if len(selected_candidates) > 0:
                frames_with_pairs.add(i)
                for cand in selected_candidates:
                    frame_pairs.append({
                        "frame_i": i,
                        "frame_j": cand["frame_j"],
                        "T_relative": cand["T_relative"],
                        "type": "best_match",
                        "overlap": cand["overlap"],
                        "baseline": cand["baseline"],
                        "score": cand["score"],
                    })

        # Add identity transformation for any frame that did not get a match
        # so every frame still produces a pose encoding.
        T_identity = torch.eye(4, device=poses.device, dtype=poses.dtype)
        for i in range(V):
            if i not in frames_with_pairs:
                frame_pairs.append({
                    "frame_i": i,
                    "frame_j": i,  # predict self
                    "T_relative": T_identity,
                    "type": "identity",
                })

        return frame_pairs

    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from ['neighbor', 'overlap']")


# @torch.no_grad() prevents the overlap compute from accumulating into the autograd graph.
@torch.no_grad()
def _compute_frame_overlap(
    world_coords_i, world_coords_j,
    valid_mask_i, valid_mask_j,
    distance_threshold=0.05
):
    """
    Compute the 3D overlap between two frames using a voxel-based fast path.
    [Optimized fully-tensorized implementation.]
    """
    # Pull out the valid 3D points.
    points_i = world_coords_i[valid_mask_i]  # (N_i, 3)
    points_j = world_coords_j[valid_mask_j]  # (N_j, 3)

    if len(points_i) < 100 or len(points_j) < 100:
        return 0.0

    # Subsample to keep this affordable.
    if len(points_i) > 5000:
        indices = torch.randperm(len(points_i), device=points_i.device)[:5000]
        points_i = points_i[indices]
    if len(points_j) > 5000:
        indices = torch.randperm(len(points_j), device=points_j.device)[:5000]
        points_j = points_j[indices]

    # Voxelize
    voxel_size = distance_threshold
    voxels_i = (points_i / voxel_size).long()
    voxels_j = (points_j / voxel_size).long()

    # --- Optimization core ---
    # 1. Find unique voxels on the GPU directly (== set(A)).
    # dim=0 makes unique() act per row (i.e. per [vx, vy, vz] coord).
    unique_voxels_i = torch.unique(voxels_i, dim=0)
    unique_voxels_j = torch.unique(voxels_j, dim=0)

    # 2. Concat the two unique sets.
    combined_voxels = torch.cat([unique_voxels_i, unique_voxels_j], dim=0)

    # 3. Unique over the combined set (== set(A) | set(B)).
    unique_combined = torch.unique(combined_voxels, dim=0)

    # 4. Sizes of |A|, |B|, |A ∪ B|.
    len_i = unique_voxels_i.shape[0]
    len_j = unique_voxels_j.shape[0]
    len_union = unique_combined.shape[0]

    # 5. Inclusion-exclusion: |A ∩ B| = |A| + |B| - |A ∪ B|.
    len_intersection = len_i + len_j - len_union

    # 6. IoU.
    overlap_score = len_intersection / (len_union + 1e-6)

    # Single .item() call after all the math is done.
    return overlap_score.item()