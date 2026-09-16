# Project Statement

## 1. Problem Statement
Municipal solid waste management is an urgent global ecological challenge. Inefficient waste segregation at the source leads to high rates of landfill disposal, severe environmental pollution, and the loss of recyclable commodities. Manual sorting in recycling facilities is labour-intensive, hazardous, economically inefficient, and prone to human error. Developing an automated, robust, and lightweight Computer Vision system capable of classifying waste objects into recyclable and non-recyclable streams (cardboard, glass, metal, paper, plastic, and general trash) directly addresses this critical gap.

## 2. Scope of the Project
The scope of this project encompasses:
- Building an end-to-end, modular Computer Vision pipeline using PyTorch and Torchvision.
- Utilizing Transfer Learning with an ImageNet-pretrained `MobileNetV3-Small` convolutional neural network backbone to achieve high classification accuracy with low computational latency on commodity CPU hardware.
- Developing automated data acquisition, validation, stratified splitting (70% train, 15% validation, 15% test), and domain-specific data augmentations.
- Designing a deterministic, reproducible training engine with early stopping, model checkpointing, loss monitoring, and learning rate scheduling.
- Performing rigorous test-set evaluation computing multi-class Accuracy, Precision, Recall, Macro/Weighted F1-score, and generating Confusion Matrix visualizations.
- Delivering a production-ready Command-Line Interface (CLI) supporting validation, training, batch evaluation, and single-image Top-K inference with image validation and confidence heuristics.
- Operating fully on standard terminal environments without requiring external graphical user interfaces or proprietary hardware accelerators.

## 3. Target Users
1. **Recycling and Waste Management Facilities**: Operators seeking automated optical sorting solutions for conveyor belts to segregate recyclable commodities.
2. **Municipal Smart City Planners**: Implementers of automated smart trash bins that prompt citizens with correct recycling actions based on camera detection.
3. **Academic Evaluators & Computer Vision Researchers**: Students, instructors, and researchers benchmarking transfer learning efficiency, mobile CNN backbones, and classification performance on real-world waste image distributions.
4. **Embedded/Edge AI Developers**: Engineers porting vision classifiers to resource-constrained IoT devices (e.g., Raspberry Pi, Jetson Nano).

## 4. High-Level Features
- **Automated Dataset Pipeline**: One-command retrieval, extraction, corruption checking, and stratified splitting of benchmark waste datasets.
- **Robust Preprocessing & Augmentation**: Pipeline applying rotation, flipping, cropping, tensor conversion, and ImageNet standardization to prevent overfitting while isolating validation/test sets from augmentation leakage.
- **Transfer Learning with MobileNetV3-Small**: Pretrained lightweight backbone with a customized classification head for six-class waste discrimination.
- **Dual-Mode Training**: Configurable full training with AdamW and early stopping, alongside a `--quick` development mode for fast verification on CPU environments.
- **Comprehensive Academic Metrics**: Automatic generation of Confusion Matrix plots, per-class classification reports, and training history CSV/curve visualizations.
- **Robust Inference System**: Validates image path, extension (.jpg, .jpeg, .png), and tensor integrity; delivers predicted class label, percentage confidence score, and optional Top-K probabilistic breakdown.
- **Reproducibility & Verification**: Pinned pseudo-random seed enforcement across Python, NumPy, and PyTorch, accompanied by an automated suite of unit tests and project health check scripts.
