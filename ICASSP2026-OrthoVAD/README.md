# OrthoVAD: Weakly Supervised Video Anomaly Detection via Prototype Orthogonality Learning

![ICASSP](https://img.shields.io/badge/ICASSP-2026-blue.svg)
![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg)

Official PyTorch implementation of the ICASSP 2026 paper:  
**"OrthoVAD: Weakly Supervised Video Anomaly Detection via Prototype Orthogonality Learning"**.

## 📝 Abstract

Weakly Supervised Video Anomaly Detection (VAD) aims to identify anomalous events using only video-level labels. However, the coarse-grained nature of these labels causes normal segments within anomalous videos to be incorrectly labeled as abnormal, leading to severe confusion between normal and abnormal features in the representation space.

To address this challenge, we propose a novel framework named **OrthoVAD**. Its core idea is to introduce a powerful global geometric constraint—prototype orthogonality—to proactively construct a structured and easily separable feature space. Through the synergistic optimization of a **Contextual Sparse Perception Module (CSPM)** and **Prototype Orthogonality Geometric Constraint (POGC)**, OrthoVAD learns highly discriminative features for weakly supervised video anomaly detection.

## 🔥 Highlights

- **Contextual Sparse Perception Module (CSPM)** for temporal feature refinement.
- **Prototype Orthogonality Geometric Constraint (POGC)** for global feature-space separation.
- Joint optimization with **MIL loss**, **gate sparsity loss**, and **prototype orthogonality loss**.
- Experiments on **UCF-Crime** and **ShanghaiTech** using CLIP-based video features.
- Simple PyTorch implementation for academic reference and reproducibility.

## 🚀 Environment Setup

Ensure you have a Python 3.10 environment, preferably configured via Anaconda.

```bash
conda create -n orthovad python=3.10
conda activate orthovad
```

Install the required dependencies:

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm pyyaml
```

## 📁 Project Structure

The current script-style project structure is:

```text
OrthoVAD/
├── README.md
├── Dataset_sh.py
├── Dataset_ucf.py
├── eval.py
├── main.py
├── options.py
├── OrthoVAD.py
├── test.py
├── train.py
├── utils.py
├── dataset/
└── features/
```

Before uploading the repository, please avoid uploading local cache files, IDE configuration files, datasets, features, checkpoints, and experiment outputs, such as:

```text
.idea/
__pycache__/
dataset/
features/
ckpt/
result/
*.pth
*.pt
*.npy
*.pickle
*.pkl
```

## 📦 Dataset Preparation

This repository supports experiments on:

- **UCF-Crime**
- **ShanghaiTech**

Please organize the dataset annotations as follows:

```text
dataset/
├── ucf-crime/
│   ├── train_split_10crop.txt
│   ├── test_split_10crop.txt
│   └── GT/
│       ├── video_label_10crop.pickle
│       └── ucf_gt_upgate.pickle
└── shanghaitech/
    ├── train_split_10crop.txt
    ├── test_split_10crop.txt
    └── GT/
        ├── video_label_10crop.pickle
        └── frame_label.pickle
```

The extracted CLIP features can be organized separately:

```text
features/
├── UCF-Crime/
│   ├── video_001.npy
│   ├── video_002.npy
│   └── ...
└── ShanghaiTech/
    ├── video_001.npy
    ├── video_002.npy
    └── ...
```

> Note: Due to dataset licensing restrictions, this repository does not redistribute the original videos or dataset files. Please obtain the datasets from their official sources and prepare the corresponding feature files locally.

## ⚙️ Main Arguments

Important arguments in `options.py`:

| Argument | Description | Default |
|---|---|---:|
| `--dataset_name` | Dataset name: `ucf-crime` or `shanghaitech` | `ucf-crime` |
| `--dataset_path` | Root path of dataset annotation files | `./dataset` |
| `--feature_dir_ucf` | Path to UCF-Crime CLIP features | `./features/UCF-Crime` |
| `--feature_dir_sh` | Path to ShanghaiTech CLIP features | `./features/ShanghaiTech` |
| `--lr` | Learning rate | `0.0001` |
| `--batch_size` | Batch size used by the dataloader | `1` |
| `--sample_size` | Number of normal/anomalous videos sampled per iteration | `30` |
| `--max-seqlen` | Maximum sequence length | `300` |
| `--k` | Top-K divisor used in MIL loss | `8` |
| `--lambda_mil` | Weight of MIL loss | `1.0` |
| `--lambda_ortho` | Weight of prototype orthogonality loss | `0.02` |
| `--lambda_gate` | Weight of gate sparsity loss | `1.0` |

## 🏋️ Training

Train OrthoVAD on UCF-Crime:

```bash
python main.py \
  --dataset_name ucf-crime \
  --dataset_path ./dataset \
  --feature_dir_ucf ./features/UCF-Crime \
  --device 0
```

Train OrthoVAD on ShanghaiTech:

```bash
python main.py \
  --dataset_name shanghaitech \
  --dataset_path ./dataset \
  --feature_dir_sh ./features/ShanghaiTech \
  --device 0
```

To use a pretrained checkpoint:

```bash
python main.py \
  --dataset_name ucf-crime \
  --dataset_path ./dataset \
  --feature_dir_ucf ./features/UCF-Crime \
  --pretrained_ckpt ./ckpt/orthovad_ucf_crime.pth \
  --device 0
```

## 🧪 Evaluation During Training

The training script evaluates the model every `--snapshot` iterations. Evaluation results are saved to:

```text
result/OrthoVAD/<timestamp>/result.txt
```

Anomaly score visualizations can be enabled with:

```bash
python main.py \
  --dataset_name ucf-crime \
  --dataset_path ./dataset \
  --feature_dir_ucf ./features/UCF-Crime \
  --plot 1 \
  --device 0
```

## 📊 Main Results

Frame-level AUC (%) on benchmark datasets:

| Method | Feature | ShanghaiTech | UCF-Crime |
|---|---:|---:|---:|
| OrthoVAD | CLIP ViT-B/16 | **98.13** | **88.44** |

## 🧩 Method Overview

OrthoVAD consists of three main optimization components:

### Multiple Instance Learning Loss

The MIL loss performs weakly supervised instance-level anomaly discrimination based on video-level labels.

### Contextual Sparse Perception Module

CSPM first encodes temporal context with a bidirectional GRU, then applies temporal attention and sparse gating to highlight discriminative video segments.

### Prototype Orthogonality Geometric Constraint

POGC constructs normal and abnormal prototypes from refined features and encourages them to become orthogonal in the feature space, improving global separability between normal and abnormal representations.

The final objective is:

```text
L = λ_mil L_mil + λ_ortho L_ortho + λ_gate L_gate
```

## 📌 Pretrained Checkpoints

Pretrained checkpoints are not included in this repository by default. If released later, they can be organized as follows:

```text
ckpt/
├── orthovad_ucf_crime_clip_vitb16.pth
└── orthovad_shanghaitech_clip_vitb16.pth
```

## 📚 Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@inproceedings{zhu2026orthovad,
  title={OrthoVAD: Weakly Supervised Video Anomaly Detection via Prototype Orthogonality Learning},
  author={Zhu, Tao and Liu, Yue and Luo, Kaiwen and Huang, Longjie and Tu, Xinyi and Cheng, Yuheng and Li, Shiyu and Yu, Qi and Tu, Hao and Shu, Lei},
  booktitle={ICASSP},
  year={2026}
}
```

## 🙏 Acknowledgements

We sincerely thank the researchers and open-source contributors in the field of weakly supervised video anomaly detection. This work is inspired by prior studies on multiple instance learning, temporal feature modeling, feature representation learning, and CLIP-based video anomaly detection.

In particular, we acknowledge the contributions of previous representative works, including but not limited to Sultani et al., RTFM, MIST, S3R, VadCLIP, TPWGN, and STPrompt. Their research has provided important foundations and valuable insights for the development of weakly supervised video anomaly detection.

We also thank the authors and maintainers of the UCF-Crime and ShanghaiTech datasets, as well as the developers of PyTorch and related open-source libraries.

## 📄 License

No formal license is specified at this stage. The code is provided for academic reference and reproducibility.

## 📬 Contact

For questions or discussions, please open an issue or contact the authors.
