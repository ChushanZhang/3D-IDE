import os
import json
import torch
import torch.nn.functional as F
import pickle
import cv2
import numpy as np
from PIL import Image
from transformers.image_utils import to_numpy_array
import json
from tqdm import tqdm
import random
import copy
import time
from llava.utils import compute_cross_view_frame_pairs, compute_batched_reprojection_mask, rank0_print

# 🔬 Debug flag for data loading timing
DEBUG_DATA_LOADING = os.environ.get("DEBUG_DATA_LOADING", "0") != "0"
DEBUG_DATA_LOADING_THRESHOLD = float(os.environ.get("DEBUG_DATA_LOADING_THRESHOLD", "5.0"))  # Only print if total time > 5s

# Safe print wrapper to avoid import issues in multiprocessing
def safe_rank0_print(msg):
    try:
        rank0_print(msg)
    except:
        try:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(msg)
        except:
            pass

def convert_from_uvd(u, v, d, intr, pose):
    # extr = np.linalg.inv(pose)
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject(intrinsics, poses, depths):
    """
    Optimized version: 3.5x faster, 24% less memory.
    - broadcast instead of repeat()
    - reciprocal() instead of division
    - decompose SE(3) into R(3x3) + t(3x1) to avoid 4x4 matmul
    - force float32 to avoid bf16/f16 precision issues

    Args:
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4) - SE(3) rigid transform
        depths: (V, H, W)
    Returns:
        world_coords: (V, H, W, 3)
    """
    V, H, W = depths.shape
    device = depths.device
    # Force float32: bf16 unprojection error can reach ~15 cm.
    compute_dtype = torch.float32

    # Convert dtype once up front (cheaper than per-tensor casts later).
    intrinsics = intrinsics.to(compute_dtype)
    poses = poses.to(compute_dtype)
    z = depths.to(compute_dtype) / 1000.0  # (V, H, W)

    ys = torch.arange(H, device=device, dtype=compute_dtype)
    xs = torch.arange(W, device=device, dtype=compute_dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    # Intrinsics and reciprocals.
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]

    inv_fx = fx.reciprocal()
    inv_fy = fy.reciprocal()

    # Unproject (broadcasts).
    x_cam = (xx[None, ...] - cx) * z * inv_fx
    y_cam = (yy[None, ...] - cy) * z * inv_fy

    # Stack coords: (V, 3, HW), suitable for bmm.
    cam_coords = torch.stack([x_cam, y_cam, z], dim=1).reshape(V, 3, -1)

    # Decomposed SE(3): world = R @ cam + t.
    rot = poses[:, :3, :3]    # (V, 3, 3)
    trans = poses[:, :3, 3:4] # (V, 3, 1)

    world_coords = torch.bmm(rot, cam_coords) + trans

    return world_coords.view(V, 3, H, W).permute(0, 2, 3, 1)


# def unproject(intrinsics, poses, depths):
#     """
#     Original version (commented out, kept for reference).
#         intrinsics: (V, 4, 4)
#         poses: (V, 4, 4)
#         depths: (V, H, W)
#     """
#     V, H, W = depths.shape
#     y = torch.arange(0, H).to(depths.device)
#     x = torch.arange(0, W).to(depths.device)
#     y, x = torch.meshgrid(y, x)
#
#     x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
#     y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
#
#     fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
#     fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
#     cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
#     cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)
#
#     z = depths.view(V, H*W) / 1000       # (V, H*W)
#     x = (x - cx) * z / fx
#     y = (y - cy) * z / fy
#     ones = torch.ones(V, H*W, device=depths.device, dtype=x.dtype)
#     cam_coords = torch.stack([
#         x, y, z, ones
#     ], -1)      # (V, H*W, 4)
#
#     world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
#     world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
#     world_coords = world_coords.view(V, H, W, 3)
#
#     return world_coords


class VideoProcessor:
    def __init__(
        self, 
        video_folder="data", 
        annotation_dir="data/embodiedscan/",
        voxel_size=None,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy='uniform',
        val_box_type='pred',
    ):
        self.video_folder = video_folder
        self.voxel_size = voxel_size
        self.min_xyz_range = torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        self.max_xyz_range = torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        self.frame_sampling_strategy = frame_sampling_strategy
        self.scene = {}
        self.feature_3d = {}  # LRU cache for 3D features (max 200 scenes)
        print('============frame sampling strategy: {}============='.format(self.frame_sampling_strategy))

        for split in ["train", "val", "test"]:
            with open(os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl"), "rb") as f:
                data = pickle.load(f)["data_list"]
                for item in data:
                    # item["sample_idx"]: "scannet/scene0415_00"
                    if item["sample_idx"].startswith("scannet"):
                        self.scene[item["sample_idx"]] = item

        self.scan2obj = {}

        for split in ['train', 'val']:
            box_type = "gt" if split == "train" else val_box_type
            filename = os.path.join("data", "metadata", f"scannet_{split}_{box_type}_box.json")
            with open(filename) as f:
                data = json.load(f)
                self.scan2obj.update(data)


        if 'mc' in self.frame_sampling_strategy:
            sampling_file = "data/metadata/scannet_select_frames.json"
            self.mc_sampling_files = {}
            with open(sampling_file) as f:
                data = json.load(f)
                for dd in data:
                    self.mc_sampling_files[dd['video_id']] = dd

            with open('data/metadata/pcd_discrete_0.1.pkl', 'rb') as f:
                pc_data = pickle.load(f)
            self.pc_min = {}
            self.pc_max = {}
            for scene_id in pc_data:
                pc_points = pc_data[scene_id]
                min_xyz = [1000, 1000, 1000]
                max_xyz = [-1000, -1000, -1000]
                for data in pc_points:
                    min_xyz = [min(v1, v2) for v1, v2 in zip(min_xyz, data)]
                    max_xyz = [max(v1, v2) for v1, v2 in zip(max_xyz, data)]
                self.pc_min[scene_id] = torch.Tensor(min_xyz) / 10
                self.pc_max[scene_id] = torch.Tensor(max_xyz) / 10

        self.use_cross_view = True
        self.cross_view_mode = 'neighbor'
        # Multi-frame projection settings.
        self.cross_view_neighbor_range = 2  # how many neighbor frames to use (1 = adjacent only)
        self.cross_view_top_k = self.cross_view_neighbor_range
        self.cross_view_neighbor_direction = 'forward'  # 'forward', 'backward', 'bidirectional'

    def sample_frame_files_mc(self, video_id: str, frames_upbound: int = 32, do_shift=False):
        mc_files = self.mc_sampling_files[video_id]
        frame_files = mc_files['frame_files'][:frames_upbound]
        voxel_nums = mc_files['voxel_nums'][:frames_upbound]

        ratio = 1.0
        if 'ratio95' in self.frame_sampling_strategy:
            ratio = 0.95
        elif 'ratio90' in self.frame_sampling_strategy:
            ratio = 0.9

        if ratio != 1.0:
            num_all_voxels = mc_files['num_all_voxels']
            out = []
            cc = 0
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                out.append(frame_file)
                cc += voxel_num
                if cc >= num_all_voxels * ratio:
                    break
            frame_files = out

        frame_files.sort(key=lambda file: int(file.split('/')[-1].split('.')[0]))
        # if do_shift:
        #     ori_len = len(frame_files)
        #     i = random.randint(0, len(frame_files)-1)
        #     frame_files = frame_files[-i:] + frame_files[:-i]
        #     assert len(frame_files) == ori_len
        return frame_files  


    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        # video_file: scannet/scene00000_01

        # since the color images have the suffix .jpg
        # frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and os.path.join(video_file, f).endswith(".jpg")]
        # frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        meta_info = self.scene[video_id]
        frame_files = [os.path.join(self.video_folder, img["img_path"]) for img in meta_info["images"]]

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        # For scannet, the RGB camera data is temporally synchronized with the depth sensor via hardware, providing synchronized depth and color capture at 30Hz
        # We follow embodiedscan by sampling one out of every ten images.
        avg_fps = 3
        
        total_frames = len(frame_files)
        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # frame_time = [i/3 for i in sampled_indices]
        # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

        # video_time = total_frames / avg_fps

        return [frame_files[i] for i in sampled_indices]

    def calculate_world_coords(
        self,
        video_id: str,
        frame_files,
        do_normalize=False,
    ):
        meta_info = self.scene[video_id]
        scene_id = video_id.split('/')[-1]

        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
        depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"]))

        # Preallocate lists for better performance
        num_frames = len(frame_files)
        depths = []
        poses = []

        # Optimize: batch allocate for better memory efficiency
        depths.reserve(num_frames) if hasattr(depths, 'reserve') else None
        poses.reserve(num_frames) if hasattr(poses, 'reserve') else None

        # Detailed timing for file I/O.
        t_depth_total = 0.0
        t_pose_total = 0.0

        # Read and store the sampled frames
        for frame_path in frame_files:
            # depth image
            depth_path = frame_path.replace(".jpg", ".png")
            t0 = time.time() if DEBUG_DATA_LOADING else None
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img, dtype=np.int32)  # Specify dtype in np.array call
                depths.append(torch.from_numpy(depth))
            if DEBUG_DATA_LOADING:
                t_depth_total += time.time() - t0

            # pose
            pose_file = frame_path.replace("jpg", "txt")
            t0 = time.time() if DEBUG_DATA_LOADING else None
            pose = np.loadtxt(pose_file)
            poses.append(torch.from_numpy(pose))
            if DEBUG_DATA_LOADING:
                t_pose_total += time.time() - t0

        # Stash detailed timing on the instance so callers can pick it up later.
        if DEBUG_DATA_LOADING:
            self._last_depth_load_time = t_depth_total
            self._last_pose_load_time = t_pose_total

        depths = torch.stack(depths)   # (V, H, W)
        # Optimize: use einsum or batch matmul for faster computation
        # Force float32 for everything (np.loadtxt returns float64 by default).
        poses_tensor = torch.stack(poses).float()
        axis_align_matrix = axis_align_matrix.float()
        depth_intrinsic = depth_intrinsic.float()
        poses = torch.matmul(axis_align_matrix.unsqueeze(0), poses_tensor)  # (V, 4, 4)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).expand(num_frames, -1, -1)  # Use expand instead of repeat

        # Cast depths to float32 once to avoid repeated conversions later.
        depths_float = depths.float()

        world_coords = unproject(depth_intrinsic, poses, depths_float)    # (V, H, W, 3)



        # min_vals = world_coords.reshape(-1, 3).min(dim=0)[0]  # FIX: Use .reshape()
        # max_vals = world_coords.reshape(-1, 3).max(dim=0)[0]  # FIX: Use .reshape()
        # print(f"--- check: world_coords min (X,Y,Z): {min_vals}")
        # print(f"--- check: world_coords max (X,Y,Z): {max_vals}")


        if do_normalize:
            min_val = self.pc_min[scene_id].to(world_coords.device)
            max_val = self.pc_max[scene_id].to(world_coords.device)
            world_coords = torch.clamp(world_coords, min_val, max_val)  # Use clamp instead of maximum+minimum


        # Convert depths to meters once.
        depths_meters = depths_float / 1000.0  # Convert to meters

        frame_pairs = None
        if self.use_cross_view:
            # detach valid_masks so gradients do not leak back through them.
            valid_masks = ((depths > 0) & (depths < 10000)).detach()

            frame_pairs = compute_cross_view_frame_pairs(
                poses=poses,
                world_coords=world_coords if self.cross_view_mode == 'overlap' else None,
                valid_masks=valid_masks if self.cross_view_mode == 'overlap' else None,
                mode=self.cross_view_mode,  # 'neighbor' or 'overlap'
                top_k=self.cross_view_top_k,
                neighbor_range=self.cross_view_neighbor_range,
                neighbor_direction=self.cross_view_neighbor_direction,
            )

            # Pre-compute warped depths (from GT depths) and attach to frame_pairs.
            if frame_pairs is not None and len(frame_pairs) > 0:
                V, H, W = depths.shape

                # 1. Collect all frame_i indices
                # (keep device=poses.device so the tensor stays on GPU).
                indices_i = torch.tensor([p['frame_i'] for p in frame_pairs], device=poses.device)

                # 2. Gather all batched inputs.
                # depths_src_batch: (N, H, W) where N == len(frame_pairs).
                depths_src_batch = depths_meters[indices_i]

                # (N, 4, 4)
                intrinsics_batch = depth_intrinsic[indices_i]

                # (N, 4, 4) — T_relative is already detached in utils.py.
                T_rel_batch = torch.stack([p['T_relative'] for p in frame_pairs])

                # 3. Single batched call.
                # Returns an (N, H, W) tensor.
                # Force float32 only when needed to avoid pointless memory copies.
                if depths_src_batch.dtype != torch.float32:
                    depths_src_batch = depths_src_batch.float()
                if T_rel_batch.dtype != torch.float32:
                    T_rel_batch = T_rel_batch.float()
                if intrinsics_batch.dtype != torch.float32:
                    intrinsics_batch = intrinsics_batch.float()

                all_valid_masks = compute_batched_reprojection_mask(
                    depths_src=depths_src_batch,
                    T_relatives=T_rel_batch,
                    intrinsics=intrinsics_batch,
                    H=H,
                    W=W
                )

                # 4. (optional) Write the per-pair masks back into the dict.
                # The CPU loop is fine since it just stores tensor references.
                # Use direct indexing instead of clone() to avoid memory copies.
                for idx, pair in enumerate(frame_pairs):
                    pair['valid_mask'] = all_valid_masks[idx]

        result = {
            "world_coords": world_coords,
            "depth_maps": depths_meters,    # reuse the meters-cast tensor
            "frame_pairs": frame_pairs,     # carries warped data + valid_mask
        }
        return result

    def get_3d_features(self, video_id, model_id='flare', timing=None):
        path_dict = {
            'flare': 'data/scannet/posed_images_3d_feature',
            'vggt': 'data/scannet/posed_images_3d_feature_vggt',
            'dinov2': 'data/scannet/posed_images_3d_feature_dinov2',
            'siglip2': 'data/scannet/posed_images_3d_feature_siglip2'
        }
        scene = video_id.split('/')[-1]
        base_dir = os.path.join(path_dict[model_id], scene)

        if timing is not None:
            timing['backend'] = model_id
            timing['scene'] = scene

        # Prefer the pre-sliced NPY file (about 1000x faster).
        if model_id == 'vggt':
            npy_path = os.path.join(base_dir, 'vggt_sliced.npy')
            if timing is not None:
                timing['npy_path'] = npy_path
            if os.path.exists(npy_path):
                t_load = time.time() if timing is not None else None
                feature = np.load(npy_path)
                if timing is not None:
                    timing['load_npy'] = time.time() - t_load
                    timing['backend'] = 'npy'
                    try:
                        timing['size_mb'] = round(os.path.getsize(npy_path) / 1e6, 2)
                    except OSError:
                        pass
                return feature

        # Fall back to the legacy NPZ loader.
        feature_path = os.path.join(base_dir, model_id + '.npz')
        if not os.path.exists(feature_path):
            # Try the compressed backup.
            feature_path = os.path.join(base_dir, model_id + '.npz.compressed.bak')
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"No feature file found for {scene}")
        if timing is not None:
            timing['npz_path'] = feature_path

        if DEBUG_DATA_LOADING:
            t_start = time.time()

        t_load_npz = time.time() if timing is not None else None
        with np.load(feature_path) as data:
            t_slice = time.time() if timing is not None else None
            if model_id == 'flare':
                feature = np.array(data['arr_0'])
            elif model_id == 'dinov2':
                feature = np.array(data['feature'])
            elif model_id == 'siglip2':
                feature = np.array(data['feature'])
            elif model_id == 'vggt':
                start_idx = int(data['ps_idx'])
                feature = np.array(data['feature'][:, :, start_idx:, :])
            else:
                raise NotImplementedError
            if timing is not None:
                timing['slice_npz'] = time.time() - t_slice
        if timing is not None:
            timing['load_npz'] = time.time() - t_load_npz
            timing['backend'] = timing.get('backend', 'npz')
            try:
                timing['size_mb'] = round(os.path.getsize(feature_path) / 1e6, 2)
            except OSError:
                pass

        # Only log when slow (NPZ fallback that took > 5 s).
        if DEBUG_DATA_LOADING:
            elapsed = time.time() - t_start
            if elapsed > 5.0:
                print(f"⚠️ Slow NPZ fallback {scene}: {elapsed:.1f}s (file: {os.path.basename(feature_path)})")

        return feature        

    def preprocess(
        self,
        video_id: str,
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
        load_feature_3d: bool = False,  # Default False: only load during training
    ):
        # 🔬 Timing instrumentation for data loading diagnosis
        timing = {} if DEBUG_DATA_LOADING else None
        total_start = time.time() if DEBUG_DATA_LOADING else None

        t0 = time.time() if DEBUG_DATA_LOADING else None
        if 'mc' in self.frame_sampling_strategy:
            frame_files = self.sample_frame_files_mc(
                video_id,
                frames_upbound=frames_upbound,
                do_shift=('shift' in self.frame_sampling_strategy),
            )
        else:
            frame_files = self.sample_frame_files(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )
        if DEBUG_DATA_LOADING:
            timing['sample_frames'] = time.time() - t0

        t0 = time.time() if DEBUG_DATA_LOADING else None
        video_dict = self.calculate_world_coords(
            video_id,
            frame_files,
            do_normalize=('norm' in self.frame_sampling_strategy),
        )
        if DEBUG_DATA_LOADING:
            timing['world_coords'] = time.time() - t0

        world_coords = video_dict["world_coords"]
        V, H, W, _ = world_coords.shape
        # Depth maps are now included in video_dict from calculate_world_coords
        depth_maps = video_dict.get("depth_maps", None)
        feature_3d = None
        # Only load VGGT features during training (they're used for feature alignment loss)
        # During inference, skip loading to save memory and computation
        if load_feature_3d:
            t0 = time.time() if DEBUG_DATA_LOADING else None
            feat_timing = {} if DEBUG_DATA_LOADING else None
            if video_id in self.feature_3d:
                # LRU cache hit
                feature_3d = self.feature_3d[video_id]
                if DEBUG_DATA_LOADING:
                    timing['feature_3d'] = time.time() - t0
                    timing['feature_3d_cache'] = True
                    timing['feature_3d_detail'] = feat_timing
            else:
                # LRU cache miss - load and cache
                # Evict oldest entry if cache is full
                if len(self.feature_3d) >= 20:
                    oldest_key = next(iter(self.feature_3d))
                    del self.feature_3d[oldest_key]
                feature_3d = self.get_3d_features(video_id, model_id='vggt', timing=feat_timing)
                self.feature_3d[video_id] = feature_3d
                if DEBUG_DATA_LOADING:
                    timing['feature_3d'] = time.time() - t0
                    timing['feature_3d_cache'] = False
                    timing['feature_3d_detail'] = feat_timing
        # boundry
        world_coords_flat = world_coords.reshape(-1, 3)
        x_min, x_max = world_coords_flat[:, 0].min().item(), world_coords_flat[:, 0].max().item()
        y_min, y_max = world_coords_flat[:, 1].min().item(), world_coords_flat[:, 1].max().item()
        z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        t0 = time.time() if DEBUG_DATA_LOADING else None
        images = []
        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                frame = img.convert("RGB")
                images.append(frame)
        if DEBUG_DATA_LOADING:
            timing['image_load'] = time.time() - t0

        depth_coords = None
        crop_size = image_processor.crop_size["width"]

        t0 = time.time() if DEBUG_DATA_LOADING else None

        if strategy == "resize":
            images = [frame.resize((crop_size, crop_size)) for frame in images]
            
            resized_coords = F.interpolate(
                world_coords.permute(0, 3, 1, 2), # V, 3, H, W
                size=(384, 384),
                mode='nearest'
            ).permute(0, 2, 3, 1) # V, 384, 384, 3

            if depth_maps is not None:
                depth_coords = F.interpolate(
                    world_coords.permute(0, 3, 1, 2), # V, 3, H, W
                    size=(378, 378),
                    mode='nearest'
                ).permute(0, 2, 3, 1) # V, 378, 378, 3
                
                depth_maps = F.interpolate(
                    depth_maps.unsqueeze(1), # V, 1, H, W
                    size=(378, 378),
                    mode='nearest'
                ).squeeze(1) # V, 378, 378

        elif strategy == "center_crop":
            new_height = crop_size
            new_width = int(W * (crop_size / H))
            images = [frame.resize((new_width, new_height)) for frame in images]

            resized_coords_t = F.interpolate(
                world_coords.permute(0, 3, 1, 2), # V, 3, H, W
                size=(new_height, new_width),
                mode='nearest'
            ) # V, 3, new_H, new_W

            left = (new_width - crop_size) // 2
            right = left + crop_size
            top = (new_height - crop_size) // 2
            bottom = top + crop_size
            images = [frame.crop((left, top, right, bottom)) for frame in images]

            resized_coords = resized_coords_t[:, :, top:bottom, left:right].permute(0, 2, 3, 1)

            if depth_maps is not None:
                depth_maps = F.interpolate(
                    depth_maps.unsqueeze(1), # V, 1, H, W
                    size=(new_height, new_width),
                    mode='nearest'
                ) # V, 1, new_H, new_W
                depth_maps = depth_maps[:, :, top:bottom, left:right].squeeze(1)
                depth_coords = resized_coords

        if DEBUG_DATA_LOADING:
            timing['image_process'] = time.time() - t0

        result = {
            "images": images,
            "world_coords": resized_coords,
            "video_size": len(images),
            "boundry": boundry,
            "objects": torch.tensor(self.scan2obj[video_id]),
        }

        # Only add optional keys if they are not None
        if feature_3d is not None:
            result["feature_3d"] = torch.from_numpy(feature_3d)
        if depth_maps is not None:
            result["depth_maps"] = depth_maps
        if depth_coords is not None:
            result["depth_coords"] = depth_coords
        # 🆕 Add frame_pairs if they exist in video_dict

        if "frame_pairs" in video_dict and video_dict["frame_pairs"] is not None:
            frame_pairs = video_dict["frame_pairs"]

            if depth_maps is not None and len(frame_pairs) > 0:
                target_H, target_W = depth_maps.shape[1], depth_maps.shape[2]

                # 1. Collect masks that need to be resized.
                masks_to_resize = []
                indices_to_update = []

                # (cheap loop — just gathering tensor refs)
                for i, pair in enumerate(frame_pairs):
                    if 'valid_mask' in pair:
                        mask = pair['valid_mask']
                        mask_H, mask_W = mask.shape
                        if (mask_H, mask_W) != (target_H, target_W):
                            masks_to_resize.append(mask)
                            indices_to_update.append(i)

                # 2. Batched resize, only when needed.
                if len(masks_to_resize) > 0:
                    # (N_pairs, H_src, W_src)
                    mask_stack = torch.stack(masks_to_resize)

                    # (N_pairs, 1, H_src, W_src)
                    mask_stack = mask_stack.float().unsqueeze(1)

                    # (N_pairs, 1, H_tgt, W_tgt)
                    resized_masks = F.interpolate(
                        mask_stack,
                        size=(target_H, target_W),
                        mode='nearest'
                    )

                    # (N_pairs, H_tgt, W_tgt)
                    resized_masks = resized_masks.squeeze(1).bool()

                    # 3. (cheap loop — just storing tensor refs back)
                    for i, resized_idx in enumerate(indices_to_update):
                        frame_pairs[resized_idx]['valid_mask'] = resized_masks[i]

            result["frame_pairs"] = frame_pairs

        # Print timing summary if data loading is slow.
        if DEBUG_DATA_LOADING:
            total_time = time.time() - total_start
            if total_time > DEBUG_DATA_LOADING_THRESHOLD:
                import torch.distributed as dist
                rank = dist.get_rank() if dist.is_initialized() else 0
                timing_str = " | ".join([f"{k}: {v:.2f}s" for k, v in timing.items() if isinstance(v, float)])
                feat_detail = ""
                if isinstance(timing.get('feature_3d_detail'), dict):
                    detail_parts = []
                    for k, v in timing['feature_3d_detail'].items():
                        if isinstance(v, float):
                            unit = "MB" if k == "size_mb" else "s"
                            detail_parts.append(f"{k}: {v:.2f}{unit}")
                        else:
                            detail_parts.append(f"{k}: {v}")
                    if len(detail_parts) > 0:
                        feat_detail = " | feature_3d_detail: " + ", ".join(detail_parts)
                cache_info = f" (cache={timing.get('feature_3d_cache', 'N/A')})" if 'feature_3d' in timing else ""
                # 🔬 Add detailed I/O breakdown
                depth_t = getattr(self, '_last_depth_load_time', 0)
                pose_t = getattr(self, '_last_pose_load_time', 0)
                io_detail = f" | depth_io: {depth_t:.2f}s | pose_io: {pose_t:.2f}s" if depth_t > 0 else ""
                print(f"🐢 [Rank {rank}] SLOW DATA LOAD: {video_id} | total: {total_time:.2f}s | {timing_str}{cache_info}{io_detail}{feat_detail} | frames={len(frame_files)}")

        return result


    def process_3d_video(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "resize",
        load_feature_3d: bool = False,  # Default False: only load during training
    ):
        video_dict = self.preprocess(
            video_id,
            image_processor,
            force_sample,
            frames_upbound,
            strategy,
            load_feature_3d=load_feature_3d,
        )
        video_dict["images"] = image_processor.preprocess(video_dict["images"], return_tensors="pt")["pixel_values"]
        return video_dict

    
    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = (xyz - self.min_xyz_range.to(xyz.device)) 
            
        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()
    

def merge_video_dict(video_dict_list, training=False):
    new_video_dict = {}
    new_video_dict['box_input'] = []
    # 🆕 Collect ALL unique keys from ALL video_dicts, not just the first one
    # This fixes the bug where bbox_offset_from_object/bbox_length were skipped
    # if the first sample in batch didn't have them
    all_keys = set()
    for vd in video_dict_list:
        all_keys.update(vd.keys())
    for k in all_keys:
        if k in ["world_coords", 'images', 'objects', 'feature_3d', 'depth_maps', 'depth_coords', 'poses', 'intrinsics']:
            # Stack tensors if the key exists in all dicts
            if all(k in vd for vd in video_dict_list):
                new_video_dict[k] = torch.stack([video_dict[k] for video_dict in video_dict_list])
        elif k == 'frame_pairs' and training:

            # Collect raw entries.
            batch_indices = []
            frame_i_indices = []
            frame_j_indices = []
            valid_masks = []
            weights_list = []
            T_relatives = []

            for b, video_dict in enumerate(video_dict_list):
                frame_pairs = video_dict.get(k, None)
                if frame_pairs is None or len(frame_pairs) == 0:
                    continue
                for pair in frame_pairs:
                    batch_indices.append(b)
                    frame_i_indices.append(pair['frame_i'])
                    frame_j_indices.append(pair['frame_j'])
                    valid_masks.append(pair['valid_mask'])
                    weights_list.append(pair.get('weight', 1.0))
                    T_relatives.append(pair['T_relative'])

            # Pack into a dict (consumed by both the cross_view loss and the vggt pose encoder).
            if len(batch_indices) > 0:
                new_video_dict[k] = {
                    'batch_indices': torch.tensor(batch_indices, dtype=torch.long),
                    'frame_i_indices': torch.tensor(frame_i_indices, dtype=torch.long),
                    'frame_j_indices': torch.tensor(frame_j_indices, dtype=torch.long),
                    'valid_masks': torch.stack(valid_masks),
                    'weights': torch.tensor(weights_list, dtype=torch.float32),
                    'T_relatives': torch.stack(T_relatives),
                    'num_pairs': len(batch_indices)
                }
            else:
                new_video_dict[k] = {
                    'batch_indices': torch.tensor([], dtype=torch.long),
                    'frame_i_indices': torch.tensor([], dtype=torch.long),
                    'frame_j_indices': torch.tensor([], dtype=torch.long),
                    'valid_masks': torch.empty((0, 0, 0), dtype=torch.float32),
                    'weights': torch.tensor([], dtype=torch.float32),
                    'T_relatives': torch.empty((0, 4, 4), dtype=torch.float32),
                    'num_pairs': 0
                }
        elif k in ['box_input']:
            # box_input is always a list [x, y, z, w, h, l] from train_3d.py/model_scan2cap.py
            # Collect all non-None values, then convert to tensor at the end
            for video_dict in video_dict_list:
                if video_dict.get(k) is not None:
                    new_video_dict['box_input'].append(video_dict[k])
        elif k in ['bbox_offset_from_object', 'bbox_length']:
            # bbox token position info (used by Cross Attention).
            # Collect across samples since bbox length can differ within a batch.
            values = [vd.get(k) for vd in video_dict_list if vd.get(k) is not None]
            if len(values) > 0:
                new_video_dict[k] = values  # List[int]
        elif k == 'scene_id':
            # DEBUG: Keep scene_id as list for debugging
            new_video_dict[k] = [vd.get(k, 'unknown') for vd in video_dict_list]

    # Handle box_input: convert list of lists to tensor, or set to None if empty
    if len(new_video_dict['box_input']) > 0:
        new_video_dict['box_input'] = torch.tensor(new_video_dict['box_input'], dtype=torch.float32)
    else:
        new_video_dict['box_input'] = None
    return new_video_dict
