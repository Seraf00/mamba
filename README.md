# Mamba-Enhanced Cardiac Segmentation for Portable Echocardiography

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Research Goal

Integrate **Mamba State Space Models** into SOTA segmentation architectures to enhance performance while **reducing model parameters and inference time** for portable echocardiography applications.

### Key Objectives
- Improve segmentation accuracy on echocardiography images
- Reduce computational requirements for edge deployment
- Provide clinical explainability for model predictions
- Accurate Ejection Fraction (EF) estimation

## 📊 Dataset

**CAMUS** - Cardiac Acquisitions for Multi-structure Ultrasound Segmentation

| Property | Details |
|----------|---------|
| Modality | 2D Echocardiography |
| Patients | 500 (400 train / 50 val / 50 test) |
| Views | 2-chamber (2CH), 4-chamber (4CH) |
| Frames | End-Diastolic (ED), End-Systolic (ES) |
| Half Sequences | Full cardiac cycle with GT for all frames |
| Classes | Background, LV Endocardium, LV Epicardium, Left Atrium |
| File Format | NIfTI (.nii) |
| Clinical | Ejection Fraction computation |

### Data Organization

```
CAMUS/
├── training/                     # 450 patients (patient0001-patient0450)
│   └── patient0001/
│       ├── patient0001_2CH_ED.nii            # 2-chamber End-Diastolic image
│       ├── patient0001_2CH_ED_gt.nii         # Ground truth segmentation
│       ├── patient0001_2CH_ES.nii            # 2-chamber End-Systolic image
│       ├── patient0001_2CH_ES_gt.nii         # Ground truth segmentation
│       ├── patient0001_2CH_half_sequence.nii     # Cardiac cycle (~10-20 frames)
│       ├── patient0001_2CH_half_sequence_gt.nii  # GT for ALL sequence frames!
│       ├── patient0001_4CH_ED.nii            # 4-chamber End-Diastolic image
│       ├── patient0001_4CH_ED_gt.nii         # Ground truth segmentation
│       ├── patient0001_4CH_ES.nii            # 4-chamber End-Systolic image
│       ├── patient0001_4CH_ES_gt.nii         # Ground truth segmentation
│       ├── patient0001_4CH_half_sequence.nii     # Cardiac cycle (~10-20 frames)
│       ├── patient0001_4CH_half_sequence_gt.nii  # GT for ALL sequence frames!
│       ├── Info_2CH.cfg                      # Clinical info (EF, volumes, quality)
│       └── Info_4CH.cfg                      # Clinical info for 4CH view
└── testing/                      # 50 patients (patient0401-patient0450)
```

### Half Sequences (2CH/4CH) - **WITH GROUND TRUTH!**

Each patient has half sequence files containing:
- Full cardiac cycle from ED to ES (or ES to ED)
- Multiple intermediate frames (typically 10-20 frames)
- **Ground truth segmentation for ALL frames** (`*_half_sequence_gt.nii`)

This provides **10-20x more training data** compared to using only ED/ES frames!

**Usage Recommendations:**

| Use Case | Approach | Benefit |
|----------|----------|---------|
| **Maximum Data** | `include_sequences=True` | 10-20x more samples with GT |
| **ED/ES Only** | `include_sequences=False` (default) | Standard approach |
| **Temporal Modeling** | Use sequences for temporal consistency | Smoother predictions |
| **Curriculum Learning** | Train on ED/ES first, add sequences | Progressive difficulty |

```python
from data import CAMUSDataset, CAMUSPatient

# Option 1: Use only ED/ES frames (default)
dataset = CAMUSDataset(root_dir='path/to/CAMUS', split='train')
print(f"ED/ES samples: {len(dataset)}")  # ~1600 samples (400 patients × 2 views × 2 phases)

# Option 2: Include ALL half sequence frames (10-20x more data!)
dataset_full = CAMUSDataset(
    root_dir='path/to/CAMUS',
    split='train',
    include_sequences=True  # Include all sequence frames with GT
)
print(f"With sequences: {len(dataset_full)}")  # ~15,000-25,000 samples

# Load half sequence directly
patient = CAMUSPatient('path/to/patient0001')
images, masks = patient.load_half_sequence('2CH')  # Both have shape (T, H, W)
print(f"Frames: {images.shape[0]}, with GT masks!")
```

### Image Quality Grades

CAMUS provides image quality annotations:
- **Good (2)**: High quality, clear boundaries
- **Medium (1)**: Acceptable quality
- **Poor (0)**: Low quality, unclear boundaries

```python
# Filter by quality
dataset = CAMUSDataset(
    root_dir='path/to/CAMUS',
    quality_filter=['Good', 'Medium']  # Exclude poor quality
)
```

### Data Splits (Official from Leclerc et al. TMI 2019)

| Split | Patients | ED/ES Samples | With Half Sequences* |
|-------|----------|---------------|---------------------|
| Training | 400 | 1,600 | ~20,000-32,000 |
| Validation | 50 | 200 | ~2,500-4,000 |
| Testing | 50 | 200 | ~2,500-4,000 |

*Half sequences have ~10-20 frames per view, all with ground truth segmentation.

The official splits from the CAMUS challenge are stored in `data/splits/`:
- `train.txt` - 400 patients
- `val.txt` - 50 patients  
- `test.txt` - 50 patients

Reference: Leclerc et al., "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography", IEEE TMI 2019

## 🏗️ Architecture Overview

### Base Models (9 architectures)

| Model | Description | Pretrained |
|-------|-------------|------------|
| **UNet V1** | Classic encoder-decoder with skip connections | ❌ |
| **UNet V2** | Enhanced with SE attention, attention gates, residual blocks | ❌ |
| **UNet-ResNet** | UNet with ResNet encoder (18/34/50/101/152) | ✅ |
| **DeepLab V3** | Atrous Spatial Pyramid Pooling (ASPP) | ✅ |
| **nnUNet** | Instance norm, LeakyReLU, deep supervision | ❌ |
| **GUDU** | Dense skip connections, global context | ❌ |
| **Swin-UNet** | Pure Swin Transformer architecture | ❌ |
| **TransUNet** | Hybrid CNN (ResNet) + Vision Transformer | ✅ |
| **FPN** | Feature Pyramid Network for multi-scale | ✅ |

### Mamba-Enhanced Models (10 variants)

| Model | Integration Strategy | Key Mamba Components |
|-------|---------------------|----------------------|
| **Mamba-UNet V1** | Basic | Bottleneck + Skip connections |
| **Mamba-UNet V2** | Hybrid Attention | HybridAttentionMamba, CrossMambaFusion, MultiscaleMambaBottleneck |
| **Mamba-UNet-ResNet** | Gated Fusion | GlobalContextMambaBottleneck, GatedMambaSkip |
| **Mamba-DeepLab** | ASPP Branch | MambaASPP (parallel Mamba branch), decoder Mamba |
| **Mamba-nnUNet** | Dual-Path | DualPathMambaBottleneck, deep supervision |
| **Mamba-GUDU** | Channel Attention | MambaChannelAttention, MambaGlobalContext |
| **Mamba-Swin-UNet** | Gated Swin-Mamba | SwinMambaBlock with gated attention+Mamba fusion |
| **Mamba-TransUNet** | ViT Enhancement | MambaViTBlock, cascaded Mamba upsampler |
| **Mamba-FPN** | Multi-Scale | MambaLateralConnection, MambaTopDownPath |
| **Pure-Mamba-UNet** | All-Mamba | BidirectionalMamba, minimal convolutions |

### Mamba Variants Supported

All Mamba-enhanced models support three SSM variants via `mamba_type` parameter:

| Variant | Description | Use Case |
|---------|-------------|----------|
| `'mamba'` | Original Mamba (S6) selective state space | General purpose |
| `'mamba2'` | Mamba-2 with State Space Duality (SSD) | Faster training |
| `'vmamba'` | Visual Mamba with 2D cross-scan (SS2D) | 2D spatial modeling |

### Model Factory Usage

```python
from models import get_model, list_models

# List all available models
print(list_models())
# Output: ['unet_v1', 'unet_v2', 'unet_resnet', 'deeplab_v3', 'nnunet', 'gudu',
#          'swin_unet', 'transunet', 'fpn', 'mamba_unet_v1', 'mamba_unet_v2', ...]

# Create a model
model = get_model(
    name='mamba_unet_v1',
    in_channels=1,
    num_classes=4,
    mamba_type='vmamba',  # 'mamba', 'mamba2', or 'vmamba'
    d_state=16,
    pretrained=False
)

# Create base model (no Mamba)
base_model = get_model('unet_v1', in_channels=1, num_classes=4)
```

## 📁 Project Structure

```
Paper1/
├── configs/                      # Configuration files
│   ├── model_configs.yaml        # Model hyperparameters
│   ├── training_configs.yaml     # Training settings
│   └── experiment_configs.yaml   # Experiment definitions
│
├── data/                         # Data handling
│   ├── __init__.py
│   ├── camus_dataset.py          # CAMUS dataset loader
│   ├── transforms.py             # Data augmentation
│   ├── dataloader.py             # DataLoader utilities
│   └── splits/                   # Official CAMUS splits
│       ├── train.txt             # 400 patients
│       ├── val.txt               # 50 patients
│       └── test.txt              # 50 patients
│
├── models/                       # All model architectures
│   ├── __init__.py
│   │
│   ├── base/                     # Base models (no Mamba)
│   │   ├── __init__.py
│   │   ├── unet_v1.py            # Classic UNet
│   │   ├── unet_v2.py            # Enhanced UNet with attention
│   │   ├── unet_resnet.py        # UNet with ResNet encoder
│   │   ├── deeplab_v3.py         # DeepLabV3 with ASPP
│   │   ├── nnunet.py             # nnUNet architecture
│   │   ├── gudu.py               # Dense UNet with global context
│   │   ├── swin_unet.py          # Swin Transformer UNet
│   │   ├── transunet.py          # CNN-Transformer hybrid
│   │   └── fpn.py                # Feature Pyramid Network
│   │
│   ├── mamba_enhanced/           # Mamba-integrated models
│   │   ├── __init__.py
│   │   ├── mamba_unet_v1.py      # Basic Mamba integration
│   │   ├── mamba_unet_v2.py      # Hybrid attention-Mamba
│   │   ├── mamba_unet_resnet.py  # Gated Mamba with ResNet
│   │   ├── mamba_deeplab.py      # MambaASPP integration
│   │   ├── mamba_nnunet.py       # Dual-path Mamba nnUNet
│   │   ├── mamba_gudu.py         # Channel attention Mamba
│   │   ├── mamba_swin_unet.py    # Swin-Mamba hybrid
│   │   ├── mamba_transunet.py    # ViT-Mamba hybrid
│   │   ├── mamba_fpn.py          # Multi-scale Mamba FPN
│   │   └── pure_mamba_unet.py    # Pure SSM architecture
│   │
│   └── modules/                  # Reusable Mamba modules
│       ├── __init__.py
│       ├── mamba_block.py        # Core: MambaBlock, Mamba2Block, VMMambaBlock
│       ├── mamba_encoder.py      # MambaEncoder, MambaEncoderStage
│       ├── mamba_decoder.py      # MambaDecoder, MambaDecoderStage
│       ├── mamba_bottleneck.py   # 5 bottleneck variants
│       ├── mamba_skip.py         # 4 skip connection variants
│       └── mamba_hybrid.py       # HybridAttentionMamba, MambaAttentionBlock
│
├── explainability/               # Model interpretability
│   ├── __init__.py
│   ├── gradcam.py                # Grad-CAM and variants
│   ├── attention_maps.py         # Attention visualization
│   ├── mamba_state_viz.py        # SSM state visualization
│   ├── feature_maps.py           # Intermediate feature extraction
│   ├── uncertainty.py            # Uncertainty estimation
│   └── clinical_report.py        # Clinical explainability report
│
├── metrics/                      # Evaluation metrics
│   ├── __init__.py
│   ├── segmentation_metrics.py   # Dice, IoU, HD95, ASSD
│   ├── ejection_fraction.py      # EF computation, Bland-Altman
│   └── efficiency_metrics.py     # Parameters, FLOPs, latency, portability
│
├── training/                     # Training infrastructure
│   ├── __init__.py
│   ├── trainer.py                # Training loop
│   ├── losses.py                 # Loss functions (Dice, CE, Focal, Boundary)
│   ├── scheduler.py              # LR schedulers (cosine, polynomial)
│   └── callbacks.py              # EarlyStopping, Checkpointing, Logging
│
├── inference/                    # Inference and benchmarking
│   ├── __init__.py
│   ├── predictor.py              # Model inference (TTA, uncertainty)
│   └── postprocessing.py         # Mask postprocessing (morphology, CRF)
│
├── experiments/                  # Experiment management
│   ├── __init__.py
│   └── experiment_runner.py      # Automated experiments
│
├── utils/                        # Utilities
│   ├── __init__.py
│   ├── visualization.py          # Plotting utilities
│   ├── io.py                     # File I/O helpers
│   └── misc.py                   # General helpers (seeding, timing)
│
├── scripts/                      # Entry point scripts
│   ├── train.py                  # Single model training
│   ├── train_all_models.py       # Batch training (all models)
│   ├── evaluate.py               # Single model evaluation
│   ├── evaluate_all_models.py    # Batch evaluation (all models)
│   ├── explain.py                # Single model explainability
│   ├── explain_all_models.py     # Batch explainability (all models)
│   ├── benchmark.py              # Speed/memory benchmarking
│   ├── param_match.py            # Parameter-matched wider base models
│   ├── check_mamba_setup.py      # Verify Mamba/CUDA installation
│   ├── setup_colab.py            # Colab environment setup
│   └── validate_dataset.py       # Dataset integrity validation
│
├── notebooks/                    # Jupyter notebooks
│   ├── colab_training.ipynb      # Main Colab training notebook (3 sessions)
│   ├── data_exploration.ipynb
│   ├── model_comparison.ipynb
│   └── explainability_demo.ipynb
│
├── checkpoints/                  # Model checkpoints
├── results/                      # Experiment results
├── requirements.txt
└── README.md
```

## 🔬 Explainability Framework

### Why Explainability?

Medical AI requires **transparency** for clinical adoption. Our explainability framework provides:

1. **Visual explanations** - Where is the model looking?
2. **Uncertainty quantification** - How confident is the prediction?
3. **Clinical validation** - Does it align with clinical knowledge?

### Explainability Methods

| Method | Description | Output |
|--------|-------------|--------|
| **Grad-CAM** | Gradient-weighted class activation maps | Heatmap overlay |
| **Grad-CAM++** | Improved Grad-CAM with better localization | Heatmap overlay |
| **Attention Maps** | Transformer/Mamba attention visualization | Attention weights |
| **SSM State Viz** | Mamba state dynamics visualization | State evolution plots |
| **Feature Maps** | Intermediate layer activations | Feature visualizations |
| **Uncertainty** | MC Dropout / Ensemble uncertainty | Confidence maps |
| **SHAP** | SHapley Additive exPlanations | Feature importance |
| **Clinical Report** | Automated explanation generation | Text + visual report |

### Mamba-Specific Explainability

```python
from explainability import MambaStateVisualizer, GradCAM

# Visualize Mamba state evolution
visualizer = MambaStateVisualizer(model)
state_evolution = visualizer.visualize_states(image)

# Grad-CAM for segmentation
gradcam = GradCAM(model, target_layer='bottleneck.mamba')
heatmap = gradcam.generate(image, class_idx=1)  # LV Endo

# Clinical report
from explainability import ClinicalReport
reporter = ClinicalReport(model, pixel_spacing=1.0, device='cuda')
results = reporter.generate_report(image_ed, image_es, patient_id='P001', view='4CH', save_path='report.pdf')
```

### Uncertainty Estimation

```python
from explainability import UncertaintyEstimator

# MC Dropout uncertainty
estimator = UncertaintyEstimator(model, method='mc_dropout', n_samples=20)
result = estimator.predict_with_uncertainty(image)
prediction = result['prediction']        # Mean prediction across samples
uncertainty_map = result['uncertainty']   # Predictive entropy map

# Highlight uncertain regions (e.g., boundary regions)
high_uncertainty_mask = uncertainty_map > threshold
```

### Clinical Report Generation

The clinical report includes:
- **Segmentation visualization** with overlay
- **Structure measurements** (LV volumes, wall thickness)
- **Ejection Fraction** with confidence interval
- **Uncertainty map** highlighting low-confidence regions
- **Attention/Grad-CAM** showing model focus areas
- **Quality indicators** (image quality score, prediction confidence)

## 🚀 Quick Start

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GTX 1080 (8GB) | NVIDIA RTX 3090 (24GB) |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 50 GB (dataset + checkpoints) | 100 GB |
| **CUDA** | 11.8+ | 12.1+ |

**Note**: Mamba requires CUDA. CPU-only training is not supported for Mamba models but works for base models.

### Dataset Download

1. **Register** at the [CAMUS Challenge Website](https://www.creatis.insa-lyon.fr/Challenge/camus/)
2. **Download** the dataset (requires approval)
3. **Extract** to your data directory:

```bash
# Expected structure after extraction
data/CAMUS/
├── patient0001/
│   ├── patient0001_2CH_ED.nii
│   ├── patient0001_2CH_ED_gt.nii
│   ├── patient0001_2CH_ES.nii
│   ├── patient0001_2CH_ES_gt.nii
│   ├── patient0001_2CH_half_sequence.nii
│   ├── patient0001_2CH_half_sequence_gt.nii
│   ├── patient0001_4CH_ED.nii
│   ├── patient0001_4CH_ED_gt.nii
│   ├── patient0001_4CH_ES.nii
│   ├── patient0001_4CH_ES_gt.nii
│   ├── patient0001_4CH_half_sequence.nii
│   ├── patient0001_4CH_half_sequence_gt.nii
│   ├── Info_2CH.cfg
│   └── Info_4CH.cfg
├── patient0002/
│   └── ...
└── patient0500/
```

### Installation

```bash
# Clone repository
git clone https://github.com/Seraf00/mamba.git
cd mamba

# Create environment
conda create -n mamba-seg python=3.10
conda activate mamba-seg

# Install dependencies
pip install -r requirements.txt

# Install Mamba (requires CUDA)
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

### Training

```bash
# Train Mamba-UNet V1 with VM-Mamba variant
python scripts/train.py \
    --model mamba_unet_v1 \
    --mamba_type vmamba \
    --config configs/training_configs.yaml

# Train with deep supervision
python scripts/train.py \
    --model mamba_nnunet \
    --deep_supervision \
    --config configs/training_configs.yaml
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --model mamba_unet_v1 \
    --checkpoint checkpoints/best_model.pth \
    --compute_ef  # Include EF analysis
```

### Benchmarking

```bash
# Benchmark all models
python scripts/benchmark.py --all --device cuda

# Benchmark specific model
python scripts/benchmark.py \
    --model mamba_unet_v1 \
    --input_size 256 256 \
    --batch_size 1
```

### Explainability

```bash
# Generate explanations for a prediction
python scripts/explain.py \
    --model mamba_unet_v1 \
    --checkpoint checkpoints/best.pth \
    --image path/to/image.png \
    --methods gradcam attention uncertainty \
    --output results/explanations/

# Generate clinical report
python scripts/explain.py \
    --model mamba_unet_v1 \
    --checkpoint checkpoints/best.pth \
    --image path/to/image.png \
    --clinical_report \
    --output results/reports/patient_001.pdf
```

## 📈 Metrics

### Segmentation Metrics

| Metric | Description |
|--------|-------------|
| **Dice Score** | Overlap coefficient per class |
| **IoU (Jaccard)** | Intersection over Union |
| **HD95** | 95th percentile Hausdorff Distance |
| **ASSD** | Average Symmetric Surface Distance |

### Expected Results

Performance on CAMUS test set (50 patients, ED/ES frames). Results will be updated after training:

| Model | Dice (LV) | Dice (MYO) | Dice (LA) | Mean Dice | Params (M) | GFLOPs |
|-------|-----------|------------|-----------|-----------|------------|--------|
| UNet V1 | — | — | — | — | — | — |
| UNet V2 | — | — | — | — | — | — |
| nnUNet | — | — | — | — | — | — |
| **Mamba-UNet V1** | — | — | — | — | — | — |
| **Mamba-UNet V2** | — | — | — | — | — | — |
| **Pure-Mamba-UNet** | — | — | — | — | — | — |

**Hypotheses:**
- Mamba models may achieve comparable or better accuracy with fewer parameters
- Pure-Mamba-UNet targets best efficiency for edge deployment
- Half sequence training may improve Dice across all models

### Clinical Metrics

| Metric | Description |
|--------|-------------|
| **EF Correlation** | Pearson correlation with ground truth EF |
| **EF MAE** | Mean Absolute Error for EF estimation |
| **Bland-Altman** | Agreement analysis plots |
| **Clinical Grade** | Accuracy per image quality grade |

### Efficiency Metrics

| Metric | Description |
|--------|-------------|
| **Parameters** | Total trainable parameters |
| **FLOPs** | Floating point operations |
| **Inference Time** | Time per image (ms) |
| **Memory** | Peak GPU memory usage |
| **Throughput** | Images per second |

## 🧪 Experiments

### Ablation Studies

1. **Mamba Variant Comparison**: mamba vs mamba2 vs vmamba
2. **Integration Strategy**: bottleneck vs skip vs hybrid vs full
3. **Model Size**: small vs base vs large configurations
4. **D-State Impact**: SSM state dimension ablation

### Benchmark Comparisons

- Base models vs Mamba-enhanced variants
- Parameter-accuracy trade-offs
- Speed-accuracy trade-offs
- Portable device simulation (edge GPU benchmarks)

## 📚 References

### Dataset
- **CAMUS**: [Challenge Website](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- Leclerc et al., "Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography", IEEE TMI 2019

### Mamba
- **Mamba**: [GitHub](https://github.com/state-spaces/mamba) - Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- **Mamba-2**: Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", 2024
- **VM-UNet**: [GitHub](https://github.com/JCruan519/VM-UNet) - Visual Mamba UNet for Medical Image Segmentation

### Explainability
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
- **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017

## 📄 Citation

```bibtex
@article{seraf2026mamba,
  title={Mamba-Enhanced Cardiac Segmentation for Portable Echocardiography},
  author={Seraf00},
  journal={arXiv preprint},
  year={2026}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting a pull request.

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com].
