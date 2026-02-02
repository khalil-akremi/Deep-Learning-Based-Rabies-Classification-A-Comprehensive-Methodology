# Rabies Classification using Deep Learning

## 📋 Overview
Automated classification system for rabies diagnosis using deep learning on microscopic images. This project systematically compares four state-of-the-art architectures across multiple augmentation strategies and preprocessing approaches to identify the optimal configuration for rabies detection from microscopic well images.

## 🎯 Objectives
- Compare raw vs. cropped (annotated) microscopic images for classification
- Evaluate multiple data augmentation strategies
- Benchmark deep learning architectures (CNNs and Vision Transformers)
- Develop an interpretable model for clinical deployment

## 📊 Dataset
- **Total Images**: 155 microscopic images
- **Classes**: 
  - Positif: 123 images (79.4%)
  - Negatif: 32 images (20.6%)
- **Data Split**: 70% train / 15% validation / 15% test (stratified sampling)
- **Augmented Training Set**: 432 images (4× multiplication with 3 augmentation variants per image)

## 🏗️ Architecture Comparison
Four deep learning models were evaluated:
1. **EfficientNet-B0** (~5.3M parameters) ⭐ **Best Performance**
2. **EfficientNet-B2** (~9.2M parameters)
3. **VGG16** (~138M parameters)
4. **Vision Transformer (ViT-B-16)** (~86M parameters)

## 🔄 Data Augmentation Strategies
Three augmentation approaches tested:
1. **TrivialAugmentWide**: Automated augmentation policy with 14 operations
2. **Geometric & Color**: Flips, rotations, color jittering
3. **Spatial & Blur**: Affine transformations and Gaussian blur

## ⚙️ Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1×10⁻⁴ |
| **Batch Size** | 32 (16 for ViT) |
| **Max Epochs** | 25-30 |
| **Early Stopping Patience** | 10-12 epochs |
| **LR Scheduler** | ReduceLROnPlateau |
| **Loss Function** | Weighted Cross-Entropy |
| **Cross-Validation** | Stratified 3-Fold |
| **Image Size** | 224×224 pixels |

## 🏆 Best Model Results
**EfficientNet-B0 with Cropped Images + Geometric_Color Augmentation**
- **Validation Accuracy**: 100%
- **Train-Validation Gap**: 0.68% (minimal overfitting)
- **Validation Loss**: 0.0118
- **Test Set Performance**: [See test_set_results.json]

## 📈 Key Findings
- ✅ **Cropped images outperformed raw images by 2-6%** across all architectures
- ✅ **EfficientNet models showed superior generalization** compared to VGG16 and ViT
- ✅ **Geometric_Color and Spatial_Blur augmentations** were most effective
- ✅ **ViT-B-16 underperformed (92-98%)** with high variance, indicating insufficient training data for transformers
- ✅ **Proper preprocessing and augmentation selection** are critical for optimal performance

## 🗂️ Project Structure
```
rabies_classification/
├── data/
│   ├── rabies_raw_split/              # Raw images (train/val/test)
│   │   ├── train/
│   │   ├── train_TrivialAugmentWide/
│   │   ├── train_Geometric_Color/
│   │   ├── train_Spatial_Blur/
│   │   ├── val/
│   │   └── test/
│   └── rabies_cropped_split/          # Cropped images (same structure)
│
├── rabies_results/
│   ├── checkpoints/                   # Model weights (per config)
│   ├── figures/                       # All visualizations
│   │   ├── fig1_loss_curves.png
│   │   ├── fig2_accuracy_curves.png
│   │   ├── fig3_raw_vs_cropped_comparison.png
│   │   ├── fig4_performance_heatmap.png
│   │   ├── fig5_overfitting_analysis.png
│   │   └── test_set_confusion_matrix.png
│   ├── gradcam/                       # Interpretability visualizations
│   ├── tables/                        # Results in CSV format
│   ├── best_model_config.json         # Best model configuration
│   ├── test_set_results.json          # Final test metrics
│   └── BEST_MODEL_EfficientNetB0.pth  # Best model weights
│
└── notebooks/
    └── rabies_classification.ipynb    # Main training notebook
```

## 🚀 Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA-capable GPU (recommended)
Google Colab (or local Jupyter)
```

### Required Libraries
```bash
pip install torch torchvision timm
pip install scikit-learn matplotlib seaborn
pip install opencv-python pillow pandas numpy
```

### Google Drive Setup
```python
from google.colab import drive
drive.mount('/content/drive')
```

## 📝 Usage

### Step 1: Data Preparation
```python
# Run data splitting and augmentation cell
# Creates train/val/test splits with augmented training data
```

### Step 2: Training
```python
# Run complete training orchestrator
results = run_complete_training()

# Or train single configuration for testing
history = train_single_config('EfficientNet-B0', 'cropped', 'Geometric_Color')
```

### Step 3: Evaluation
```python
# Load and analyze all results
# Generates comparison tables and visualizations
```

### Step 4: Test Set Evaluation
```python
# Evaluate best model on held-out test set
# Generates final metrics and confusion matrix
```

### Step 5: Interpretability
```python
# Generate Grad-CAM visualizations
# Shows what the model focuses on during classification
```

## 📊 Results Summary

### Performance Metrics (Test Set)
| Metric | Value |
|--------|-------|
| Accuracy | [From test results] |
| Precision | [From test results] |
| Recall | [From test results] |
| F1-Score | [From test results] |
| F2-Score | [From test results] |

### Confusion Matrix
```
                Predicted
                Negatif  Positif
Actual Negatif     TN       FP
       Positif     FN       TP
```

## 🔬 Model Interpretability
Grad-CAM (Gradient-weighted Class Activation Mapping) visualizations demonstrate that the model learns biologically relevant features rather than dataset artifacts. Heatmaps highlight regions of focus during classification decisions.

## 💾 Storage Optimization
- **Progressive checkpoint cleanup** implemented to reduce storage
- Maximum storage requirement: ~5 GB (vs. ~20 GB without cleanup)
- All metrics preserved in lightweight JSON format
- Only best model checkpoint retained for deployment

## ⚡ Training Optimization
- **Persistent workers** for faster data loading
- **Prefetch factor 2** for batch pipelining
- **Adaptive batch sizing** (reduced for ViT to prevent OOM)
- **3-Fold CV instead of 5-Fold** for 40% speedup
- **Total training time**: 10-15 hours on GPU

## 🧪 Cross-Validation Strategy
**Stratified 3-Fold Cross-Validation**
- Maintains 79%/21% class ratio in each fold
- Ensures robust performance estimation
- Reduces computational cost vs. 5-fold
- Each sample validated exactly once

## 📖 Citation
```bibtex
@article{rabies_classification_2026,
  title={Deep Learning-Based Rabies Classification from Microscopic Images},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2026}
}
```

## 🤝 Contributing
This is a research project. For questions or collaboration:
- Open an issue for bugs or questions
- Contact: [your email]

## 📄 License
[Specify your license - MIT, Apache 2.0, etc.]

## 🙏 Acknowledgments
- Dataset provided by [Institution/Source]
- Pretrained models from torchvision and timm libraries
- Computational resources: Google Colab

## 📚 References
1. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (Tan & Le, 2019)
2. Vision Transformer: An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)
3. Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)

---

**Status**: ✅ Training Complete | 📊 Results Available | 🎯 Ready for Publication

**Last Updated**: February 2026
