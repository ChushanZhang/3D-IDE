# [CVPR 2026] 3D-IDE: 3D Implicit Depth Emergent

<div align="center" style="margin-bottom:2em;">
    <a href="https://arxiv.org/abs/2604.03296" target="_blank">
        <img src="https://img.shields.io/badge/3D--IDE-ArXiv-red" alt="Paper arXiv">
    </a>
    <a href="https://chushanzhang.github.io/3D-IDE" target="_blank">
        <img src="https://img.shields.io/badge/3D--IDE-Webpage-green" alt="3D-IDE Webpage">
    </a>
</div>

<div align="center" style="margin-bottom:2em;">
  <a target="_blank" href="https://chushanzhang.github.io/">Chushan Zhang</a><sup>1</sup>,
  <a target="_blank" href="#">Ruihan Lu</a><sup>2</sup>,
  <a target="_blank" href="https://hirotong.fun/">Jinguang Tong</a><sup>1</sup>,
  <a target="_blank" href="https://yikaiw.github.io/">Yikai Wang</a><sup>3*</sup>,
  <a target="_blank" href="https://users.cecs.anu.edu.au/~hongdong/">Hongdong Li</a><sup>1*</sup>
  <br>
  <strong>
    <sup>1</sup>Australian National University &nbsp;
    <sup>2</sup>The University of Queensland &nbsp;
    <sup>3</sup>Beijing Normal University
  </strong>
  <br>
  <sup>*</sup> Corresponding authors
</div>

---

<p align="center">
    <img src="assets/overall_3d-IDE.png" width="90%"><br>
    <em>Overview of the 3D-IDE framework.</em>
</p>

---

> **License notice.** This project is released under the Apache License 2.0
> (see [`LICENSE`](LICENSE)), **except** for the vendored
> [`vggt/`](vggt/) directory, which is a copy of
> [facebookresearch/vggt](https://github.com/facebookresearch/vggt) and is
> distributed under the **Creative Commons Attribution-NonCommercial 4.0
> International License** (see [`vggt/LICENSE.txt`](vggt/LICENSE.txt)).
> Any commercial use of `vggt/` (including the VGGT pretrained checkpoint)
> is **not permitted** by that license. The rest of this repository remains
> Apache-2.0 and is unaffected.

---

## Installation

**1. Clone and create environment**
```bash
git clone https://github.com/ChushanZhang/3D-IDE.git
cd 3D-IDE

conda create -n 3d-ide python=3.10
conda activate 3d-ide
pip install --upgrade pip
```

**2. Install PyTorch and dependencies**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# torch_scatter must be installed from the prebuilt wheel index BEFORE step 3.
# It is intentionally NOT listed in pyproject.toml because PEP 517 isolated
# builds cannot see torch and would fail to compile it from source.
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```

**3. Install project and extensions**
```bash
# flash-attn 2.5.8's setup.py imports `pkg_resources`, which was removed in
# setuptools >= 70. Pin an older setuptools before building flash-attn.
pip install 'setuptools<70' wheel

pip install -e ".[train]"
pip install flash-attn==2.5.8 --no-build-isolation
pip install -e transformers
```

**Troubleshooting**

- **bitsandbytes error**: If you encounter CUDA-related errors with bitsandbytes:
  ```bash
  conda install -y -c nvidia/label/cuda-12.1.0 cuda-libraries cuda-runtime cuda-nvcc
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  ```
- **`flash-attn` build fails with `ModuleNotFoundError: No module named 'pkg_resources'`**: install `setuptools<70` first (see step 3).
- **`torch_scatter` build fails with `ModuleNotFoundError: No module named 'torch'`**: you skipped the prebuilt wheel install in step 2. Run `pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.2+cu121.html` before `pip install -e ".[train]"`.

---

## Data Preparation

1) **Processed training data**
- Download from [Hugging Face](https://huggingface.co/datasets/OliverHuang1998/3DRS) and place under `data/`.
- **Re-generate Scan2Cap data** (required — align the prompt format with VG-LLM):
  ```bash
  python scripts/3d/preprocessing/process_scan2cap.py
  ```

2) **ScanNet assets**
- Create `data/scannet/` and place:
  - `posed_images/`
  - `mask/`
  - `pcd_with_object_aabbs/`

The `mask.zip` and `pcd_with_object_aabbs.tar.gz` can be downloaded from Video-3D-LLM:
- https://huggingface.co/datasets/zd11024/Video-3D-LLM_data

---

## VGGT Features (Required for Global Supervison)

1) Download [VGGT checkpoint](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) and place at:
```
VGGT_checkpoints/model.pt
```

2) Extract features (multi-GPU):
```bash
python extract_vggt_feature_multi_gpu.py
```

This writes **only** `vggt_sliced.npy` under:
```
data/scannet/posed_images_3d_feature_vggt/<scene>/vggt_sliced.npy
```

---

## Environment Variables (Optional)

You can override common paths without editing scripts:

- `OUTPUT_BASE` (default: `./data/exp/ckpt`)
- `DPT_CHECKPOINT_PATH` (or `VGGT_CHECKPOINT_PATH` / `VGGT_CHECKPOINT`)
  - default: `VGGT_checkpoints/model.pt`
- `VGGT_ROOT_DIR` (default: `data/scannet/posed_images`)
- `VGGT_SAVE_DIR` (default: `data/scannet/posed_images_3d_feature_vggt`)
- `VGGT_NUM_FRAMES` (default: `32`)
- `CUDA_HOME` (default: `/usr/local/cuda`, used by eval scripts)

Example:
```bash
export OUTPUT_BASE=/data/exp/ckpt
export DPT_CHECKPOINT_PATH=/data/VGGT_checkpoints/model.pt
export VGGT_ROOT_DIR=/data/scannet/posed_images
export VGGT_SAVE_DIR=/data/scannet/posed_images_3d_feature_vggt
export VGGT_NUM_FRAMES=32
```

---

## Model Preparation

Download **LLaVA-Video-7B-Qwen2** from Hugging Face:
- https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2

Place it at:
```
data/models/LLaVA-Video-7B-Qwen2
```

---

## Directory Layout

```
data/
├── embodiedscan/
├── metadata/
├── models/
│   └── LLaVA-Video-7B-Qwen2/
├── processed/
└── scannet/
    ├── mask/
    ├── pcd_with_object_aabbs/
    ├── posed_images/
    └── posed_images_3d_feature_vggt/
VGGT_checkpoints/
└── model.pt
```

---

## Training

Full multi-task training (all datasets):
```bash
bash scripts/3d/train/train_multi.sh [OUTPUT_BASE]
```

---

## Evaluation

Each eval script runs inference + metric evaluation:

```bash
# ScanRefer
bash scripts/3d/eval/eval_scanrefer.sh <CKPT_PATH> uniform 32

# Multi3DRefer
bash scripts/3d/eval/eval_multi3drefer.sh <CKPT_PATH> uniform 32

# Scan2Cap
bash scripts/3d/eval/eval_scan2cap.sh <CKPT_PATH> uniform 32

# ScanQA
bash scripts/3d/eval/eval_scanqa.sh <CKPT_PATH> uniform 32

# SQA3D
bash scripts/3d/eval/eval_sqa3d.sh <CKPT_PATH> uniform 32
```

---

## Acknowledgements

This codebase builds on [3DRS](https://github.com/Visual-AI/3DRS), [Video-3D LLM](https://github.com/LaVi-Lab/Video-3D-LLM), and [VGGT](https://github.com/facebookresearch/vggt). We thank the authors for releasing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang20263d,
  title={3D-IDE: 3D Implicit Depth Emergent},
  author={Zhang, Chushan and Lu, Ruihan and Tong, Jinguang and Wang, Yikai and Li, Hongdong},
  journal={arXiv preprint arXiv:2604.03296},
  year={2026}
}
```
