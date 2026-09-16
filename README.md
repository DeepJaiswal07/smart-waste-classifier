# Smart Waste Image Classification System Using Transfer Learning

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-2.8.0%2Bcpu-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-15%20passed-brightgreen)](tests/)

An end-to-end, reproducible, command-line-driven Computer Vision system for classifying solid waste images into six categories (`cardboard`, `glass`, `metal`, `paper`, `plastic`, and `trash`) using Transfer Learning with a pretrained `MobileNetV3-Small` deep neural network backbone.

---

## 1. Overview
Municipal solid waste management has become an urgent planetary challenge. Inefficient segregation at the source contaminates recyclable streams and overwhelms municipal landfills. This project implements an automated, lightweight, and robust Computer Vision classification system designed for real-time sorting and edge execution.

By leveraging transfer learning on an ImageNet-pretrained `MobileNetV3-Small` convolutional network, the system extracts rich high-level geometric, spectral, and textural representations from waste images with fewer than 1.1 million parameters, enabling high-speed CPU execution without requiring expensive GPU clusters.

---

## 2. Problem Statement
Manual sorting of post-consumer waste in recovery facilities is labor-intensive, hazardous, economically unsustainable, and prone to human inconsistency. Automated optical sorting can rapidly categorize recyclables into distinct processing streams. However, training computer vision models from scratch requires tens of thousands of labelled samples and extensive compute power. This system solves this constraint by employing **Transfer Learning**, adapting general visual feature representations learned on ImageNet to specialized waste categories with minimal compute overhead.

---

## 3. Objectives
1. **Automated Data Pipeline**: Download, verify image integrity, filter corrupted items, and perform deterministic stratified splitting (70% train, 15% validation, 15% test).
2. **Robust Augmentation & Preprocessing**: Prevent overfitting through controlled spatial perturbations while ensuring zero data leakage into validation or test sets.
3. **Transfer Learning Architecture**: Adapt `MobileNetV3-Small` with a custom classification head designed for multi-class waste discrimination.
4. **Dual-Mode Execution**: Provide a `--quick` development mode for rapid verification on commodity CPUs alongside full-scale training.
5. **Rigorous Evaluation**: Evaluate strictly on held-out test data, producing multi-class accuracy, precision, recall, F1-scores, and confusion matrix heatmaps.
6. **Production CLI**: Deliver an ergonomic, standardized command-line interface with single-image inference, Top-K probability ranking, and input validation.

---

## 4. Key Features
- **Deterministic Reproducibility**: Fixed pseudo-random seed (`seed = 42`) across Python, NumPy, and PyTorch.
- **Hardware Agnostic**: Automatic device selection defaulting to CPU when CUDA is unavailable.
- **Memory Efficient**: On-demand lazy image loading via custom PyTorch `Dataset` without loading entire datasets into RAM.
- **Strict Leakage Isolation**: Transforms and data augmentations applied strictly post-split.
- **Comprehensive Academic Metrics**: Automated generation of `metrics.json`, `classification_report.txt`, `confusion_matrix.png`, and `training_curves.png`.
- **Defensive Error Handling**: Clear validation of image extensions, existence, and binary decodability without raw stack trace dumps.
- **Integrated Test Suite**: 15 comprehensive unit tests verifying configs, transforms, models, forward passes, and inference error handling.

---

## 5. Computer Vision Pipeline

```
                       [ Input Waste Image ]
                                 │
                                 ▼
                     [ Image Validation Layer ]
                     - File existence & extension check
                     - PIL decodability verification
                                 │
                                 ▼
                   [ Preprocessing & Normalization ]
                     - Resize: 256 x 256
                     - Crop: 224 x 224 (Random for Train / Center for Val/Test)
                     - Training Augmentation: Random Flip, Rotation (15°), Color Jitter
                     - Tensor Conversion: [0, 255] -> [0.0, 1.0]
                     - ImageNet Standardization: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]
                                 │
                                 ▼
              [ Pretrained Backbone: MobileNetV3-Small ]
                     - Depthwise separable convolutions
                     - Squeeze-and-Excitation attention blocks
                     - Hardswish activations
                     - Frozen feature extractor (927,008 params)
                                 │
                                 ▼
                [ Custom Multi-Class Head ]
                     - Linear(576 -> 256)
                     - Hardswish()
                     - Dropout(p=0.2)
                     - Linear(256 -> 6 classes) (149,254 trainable params)
                                 │
                                 ▼
                   [ Softmax Probability Distribution ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ Predicted Waste Category ]    [ Top-K Confidence Scores ]
```

---

## 6. Dataset Description

The system utilizes the benchmark **TrashNet** dataset collected by Gary Thung and Mindy Yang (Stanford University CS229).

### Class Distribution (Verified Actual Counts)
| Category | Total Samples | Train Split (70%) | Val Split (15%) | Test Split (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Cardboard** | 403 | 282 | 60 | 61 |
| **Glass** | 501 | 351 | 75 | 75 |
| **Metal** | 410 | 287 | 62 | 61 |
| **Paper** | 594 | 416 | 89 | 89 |
| **Plastic** | 482 | 337 | 72 | 73 |
| **Trash** | 137 | 96 | 21 | 20 |
| **Total** | **2,527** | **1,769** | **379** | **379** |

*Note: In accordance with repository size best practices, raw images are downloaded into `data/raw/` by the CLI and excluded from git commits via `.gitignore`.*

---

## 7. Model Architecture & Transfer Learning

`MobileNetV3-Small` was selected as the optimal architecture for this domain:
- **Efficiency**: Only 1,076,262 total parameters, requiring ~10 MB disk space.
- **Inference Speed**: Low floating-point operations per second (FLOPs) enabling real-time CPU classification (~25-50ms per frame).
- **Structural Innovations**: Employs inverted residual bottleneck blocks, depthwise separable convolutions, hard-swish activation functions, and squeeze-and-excitation (SE) attention modules.

### Parameter Allocation
- **Backbone Feature Parameters (Frozen)**: 927,008
- **Classification Head Parameters (Trainable)**: 149,254
- **Total Network Parameters**: 1,076,262

---

## 8. Technology Stack
- **Programming Language**: Python 3.10+ (tested on Python 3.12.6)
- **Deep Learning Framework**: PyTorch 2.8.0+cpu & Torchvision 0.23.0+cpu
- **Computer Vision & Image I/O**: Pillow (PIL) 11.3.0
- **Scientific Computing**: NumPy 2.2.6, Pandas 2.3.2
- **Metrics & Scikit-Learn**: Scikit-Learn 1.7.2
- **Plotting & Visualization**: Matplotlib 3.10.6
- **Configuration Management**: PyYAML 6.0.3
- **Automated Testing**: PyTest 9.1.1

---

## 9. Installation & Setup

### Clone Repository
```bash
git clone https://github.com/{username}/smart-waste-classifier.git
cd smart-waste-classifier
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
*Optional editable package installation:*
```bash
pip install -e .
```

---

## 10. Execution Guide

The system provides a unified terminal interface through `python -m src.main`:

### 1. System Self-Diagnostic Validation
Verify that configuration, models, directories, and forward passes are functioning:
```bash
python -m src.main validate
```

### 2. Dataset Preparation
Download, extract, verify file integrity, and generate deterministic stratified splits:
```bash
python -m src.main prepare-data
```

### 3. Quick Development Training (CPU Friendly)
Train for 2 epochs on a small representative slice (~120 samples) to verify the pipeline in <5 seconds:
```bash
python -m src.main train --quick
```

### 4. Full Model Training
Train on the complete dataset (1,769 train, 379 validation samples) with AdamW optimizer, validation monitoring, and best-model checkpointing:
```bash
python -m src.main train
```

### 5. Test Set Evaluation
Evaluate the checkpointed model strictly on the held-out test split (379 samples):
```bash
python -m src.main evaluate
```

### 6. Single-Image Inference
Classify a single waste image with Top-K confidence rankings:
```bash
python -m src.main predict --image examples/sample_cardboard.jpg --top-k 3
```
*Example Terminal Output:*
```text
Prediction ----------
Image:           .../examples/sample_cardboard.jpg
Predicted class: cardboard
Confidence:      99.76%
Model:           mobilenet_v3_small
Device:          CPU

Top predictions:
  1. cardboard  99.76%
  2. paper      0.22%
  3. trash      0.01%
```

### 7. Run Test Suite
Execute all unit and integration tests:
```bash
pytest -q
```

### 8. Project Audit Script
Verify directory structure, mandatory files, and configuration:
```bash
python scripts/check_project.py
```

---

## 11. Actual Experimental Results

The following metrics represent actual empirical values measured on the held-out test set (379 samples) following 5 epochs of training on CPU hardware:

### Overall Performance Metrics
| Metric | Value |
| :--- | :---: |
| **Test Accuracy** | **75.46%** |
| **Macro Precision** | **0.7311** |
| **Macro Recall** | **0.7128** |
| **Macro F1-Score** | **0.7150** |
| **Weighted F1-Score** | **0.7504** |
| **Total Training Duration (CPU)** | **80.1 seconds** |
| **Best Validation Epoch** | **Epoch 5 (Loss: 0.7108)** |

### Per-Class Performance Breakdown
| Waste Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cardboard** | 0.8000 | 0.9180 | 0.8550 | 61 |
| **Glass** | 0.7237 | 0.7333 | 0.7285 | 75 |
| **Metal** | 0.7273 | 0.7869 | 0.7559 | 61 |
| **Paper** | 0.8400 | 0.7079 | 0.7683 | 89 |
| **Plastic** | 0.7125 | 0.7808 | 0.7451 | 73 |
| **Trash** | 0.5833 | 0.3500 | 0.4375 | 20 |

---

## 12. Generated Artifacts & Outputs

All execution outputs are saved under `outputs/` and `models/`:
- `models/best_model.pth`: PyTorch binary state dictionary of the optimal model checkpoint.
- `models/class_names.json`: JSON list mapping integer indices to category names.
- `models/model_metadata.json`: Architectural details, training timestamp, and epoch logs.
- `outputs/metrics.json`: JSON document storing test accuracy, precision, recall, and F1-scores.
- `outputs/classification_report.txt`: Tabular evaluation report formatted with precision, recall, and support.
- `outputs/confusion_matrix.png`: High-resolution heatmap displaying true vs. predicted classifications.
- `outputs/training_curves.png`: Dual-panel loss and accuracy progression curves over training epochs.
- `outputs/training_history.csv`: Per-epoch metric records for loss, accuracy, and learning rates.

---

## 13. Project Structure

```
smart-waste-classifier/
├── README.md                      # Complete project documentation and execution guide
├── statement.md                   # Problem statement, scope, target users, and features
├── LICENSE                        # MIT Open-Source License
├── requirements.txt               # Pinned package dependencies
├── pyproject.toml                 # PEP 518/621 package build configuration
├── config.yaml                    # Centralized project configuration file
├── .gitignore                     # Git exclusions for datasets, weights, and caches
│
├── src/                           # Main Python package source
│   ├── __init__.py
│   ├── main.py                    # Unified CLI entry point
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py             # Dataset downloader, verifier, splitter, and Dataset class
│   │   └── preprocessing.py       # Transform pipelines for train and evaluation splits
│   ├── models/
│   │   ├── __init__.py
│   │   └── model.py               # MobileNetV3-Small transfer learning architecture
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             # AdamW training loop, early stopping, and history plotting
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # Test set evaluation, metrics, and confusion matrix
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predictor.py           # Single-image predictor, Top-K, and input validation
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # YAML config loader and validator
│       ├── seed.py                # Deterministic pseudo-random seed manager
│       ├── paths.py               # Platform-agnostic relative path resolver
│       └── logging_utils.py       # Formatted console logger
│
├── scripts/                       # Shell and diagnostic scripts
│   ├── check_project.py           # Pre-evaluation audit script
│   ├── train.sh                   # Wrapper script for training
│   ├── evaluate.sh                # Wrapper script for evaluation
│   └── predict.sh                 # Wrapper script for inference
│
├── tests/                         # PyTest unit test suite
│   ├── __init__.py
│   ├── test_config.py             # Config schema and split proportion tests
│   ├── test_preprocessing.py      # Transform shapes and normalization tests
│   ├── test_model.py              # Model instantiation and forward pass tests
│   └── test_inference.py          # Input validation and predictor error handling tests
│
├── data/                          # Dataset directory (gitignored)
│   └── .gitkeep
├── models/                        # Saved model checkpoints (gitignored)
│   └── .gitkeep
├── outputs/                       # Experiment evaluation curves and logs (gitignored)
│   └── .gitkeep
├── examples/                      # Test assets
│   ├── README.md                  # Usage instructions for inference
│   └── sample_cardboard.jpg       # Sample test image
├── notebooks/                     # Supplementary documentation
│   └── README.md                  # Note on CLI-first academic workflow
└── report/                        # Academic deliverables
    └── project_report.md          # 15-section academic project report
```

---

## 14. Reproducibility & Limitations

### Deterministic Seeding
The system enforces pseudo-random seed `42` across Python `random`, `numpy.random`, and `torch.manual_seed()`. Minor floating-point divergence can occur across differing CPU instruction sets (e.g. AVX2 vs AVX-512) or CUDA backends.

### Limitations
1. **Single-Item Assumption**: The current classifier operates on single, dominant waste objects per image; it does not perform multi-object bounding box detection.
2. **Class Imbalance**: The `trash` category has only 137 samples compared to 594 for `paper`, leading to lower recall on general residual trash.
3. **Controlled Backgrounds**: The dataset features items captured against neutral white backgrounds. Performance may degrade on heavily cluttered backgrounds without background segmentation.

---

## 15. Future Enhancements
- **Multi-Object Detection**: Integrating a lightweight detector such as YOLOv8-Nano to localize and classify multiple waste objects in a conveyor stream.
- **Class Balancing Strategies**: Employing Focal Loss or weighted random sampling to boost minority class recall.
- **Edge Deployment**: Exporting the trained PyTorch model to ONNX runtime or TensorRT for low-power embedded deployment on Raspberry Pi 5 or NVIDIA Jetson Nano.

---

## 16. References
1. Howard, A., Sandler, M., Chu, G., Chen, L. C., Chen, B., Tan, M., Wang, W., Zhu, Y., Pang, R., Vasudevan, V., Le, Q. V., & Adam, H. (2019). Searching for MobileNetV3. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 1314–1324.
2. Thung, G., & Yang, M. (2016). Classification of Trash for Recyclability Status. *CS229 Project Report*, Stanford University.
3. Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 248–255.
4. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 8024–8035.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825–2830.
