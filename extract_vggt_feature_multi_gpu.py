import os
import torch
import torch.multiprocessing as mp
import numpy as np
from tqdm import tqdm
import sys

# Add vendored vggt/ to path (the outer dir contains the vggt package as a subdirectory).
current_dir = os.path.dirname(os.path.abspath(__file__))
vggt_path = os.path.join(current_dir, "vggt")
sys.path.append(vggt_path)

# -------------------------------
# Import new model and related utility functions
# -------------------------------
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def process_scenes_on_gpu(gpu_id, scene_subset, root_dir, root_save_3d_feature, num_frames_to_sample, checkpoint_path):
    """
    Process a subset of scenes on the specified GPU.

    Args:
        gpu_id: GPU index.
        scene_subset: scenes assigned to this GPU.
        root_dir: data root directory.
        root_save_3d_feature: output root directory.
        num_frames_to_sample: number of frames sampled per scene.
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    dev_capability = torch.cuda.get_device_capability(gpu_id)
    dtype = torch.bfloat16 if dev_capability[0] >= 8 else torch.float16

    model = VGGT()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    msg = model.load_state_dict(checkpoint)
    print(f'GPU {gpu_id} loading status: {msg}')
    model = model.to(device).eval()

    for scene in tqdm(scene_subset, desc=f"GPU {gpu_id}", position=gpu_id):
        scene_dir = os.path.join(root_dir, scene)
        if not os.path.isdir(scene_dir):
            continue

        scene_save_dir = os.path.join(root_save_3d_feature, scene)
        if not os.path.exists(scene_save_dir):
            os.makedirs(scene_save_dir, exist_ok=True)

        file_names = [file for file in os.listdir(scene_dir) if file.endswith('.jpg')]
        file_names.sort()
        total_frames = len(file_names)
        if total_frames == 0:
            continue

        sampled_indices = np.linspace(0, total_frames - 1, num=num_frames_to_sample, dtype=int)
        sampled_file_list = [os.path.join(scene_dir, file_names[i]) for i in sampled_indices]

        images = load_and_preprocess_images(sampled_file_list)
        images = images.to(device, non_blocking=True).to(dtype).unsqueeze(0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
                aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Save the pre-sliced NPY only (matches the logic of scripts/convert_npz_to_npy.py).
        try:
            feature = aggregated_tokens_list[-1].cpu().numpy()
            ps_idx_np = ps_idx.cpu().numpy() if isinstance(ps_idx, torch.Tensor) else ps_idx
            start_idx = int(ps_idx_np) if np.isscalar(ps_idx_np) else int(ps_idx_np[0])
            sliced_feature = feature[:, :, start_idx:, :]
            npy_path = os.path.join(scene_save_dir, 'vggt_sliced.npy')
            np.save(npy_path, sliced_feature)

            # Remove any stale npz files so only vggt_sliced.npy remains.
            npz_path = os.path.join(scene_save_dir, 'vggt.npz')
            npz_bak_path = npz_path + '.compressed.bak'
            for p in (npz_path, npz_bak_path):
                if os.path.exists(p):
                    os.remove(p)
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning: failed to save vggt_sliced.npy for {scene}: {e}")

        del images, aggregated_tokens_list, ps_idx, feature, ps_idx_np
        torch.cuda.empty_cache()


def main():
    root_dir = os.environ.get("VGGT_ROOT_DIR", "data/scannet/posed_images")
    root_save_3d_feature = os.environ.get(
        "VGGT_SAVE_DIR", "data/scannet/posed_images_3d_feature_vggt"
    )
    checkpoint_path = os.environ.get(
        "VGGT_CHECKPOINT_PATH",
        os.environ.get("VGGT_CHECKPOINT", "VGGT_checkpoints/model.pt"),
    )
    if not os.path.exists(root_save_3d_feature):
        os.makedirs(root_save_3d_feature)
        print("create dir", root_save_3d_feature)

    scene_list = [s for s in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, s))]
    scene_list.sort()
    num_frames_to_sample = int(os.environ.get("VGGT_NUM_FRAMES", "32"))

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPU detected!")
        return

    print(f"Detected {num_gpus} GPU(s); using all of them.")
    print(f"Total {len(scene_list)} scene(s) to process.")

    scenes_per_gpu = np.array_split(scene_list, num_gpus)

    for i, scenes in enumerate(scenes_per_gpu):
        print(f"GPU {i}: assigned {len(scenes)} scene(s)")

    mp.set_start_method('spawn', force=True)
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_scenes_on_gpu,
            args=(
                gpu_id,
                scenes_per_gpu[gpu_id],
                root_dir,
                root_save_3d_feature,
                num_frames_to_sample,
                checkpoint_path,
            )
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All GPUs done.")

if __name__ == '__main__':
    main()
