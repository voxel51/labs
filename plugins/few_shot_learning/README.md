# Few-Shot Learning Plugin

Interactive few-shot learning panel for FiftyOne that lets you train classifiers by labeling just a few positive and negative samples.

## Features

- **Multiple model types**: Choose from 5 different classifiers
- **Interactive labeling**: Select samples in the grid and label them with one click
- **Iterative refinement**: Train, review predictions, add more labels, repeat
- **Fast inference**: Uses PyTorch DataLoaders for efficient batch processing

## Supported Models

| Model | Description | Best For |
|-------|-------------|----------|
| **LinearSVMModel** | Fast linear SVM classifier | General use, large datasets |
| **RocchioPrototypeModel** | Centroid-based prototype classifier | Balanced data, interpretable |
| **NCAMetricLearningModel** | Neighborhood Component Analysis | Learning distance metrics |
| **LMNNMetricLearningModel** | Large Margin Nearest Neighbor | Margin-based metric learning |
| **GraphLabelPropagationModel** | Semi-supervised label propagation | Using unlabeled data structure |

## Prerequisites

The plugin ships with self-contained model implementations. Required Python dependencies are:

- `numpy`
- `scikit-learn`
- `torch`

## Installation

```bash
fiftyone plugins download \
    https://github.com/voxel51/labs \
    --plugin-names @51labs/few_shot_learning
```

## Usage

### Quick Start

Run the demo script to launch with a sample dataset:

```bash
cd /path/to/fiftyone-labs
python plugins/few_shot_learning/run_demo.py
```

### Manual Usage

1. Launch FiftyOne with your dataset
2. Open the **Few-Shot Learning** panel from the + menu
3. Select your embedding field and model type
4. Click **Start Session**
5. Select samples in the grid and click **Label Positive** or **Label Negative**
6. Click **Train & Label Dataset** to run predictions
7. Review results, add more labels, and iterate
8. Click **Tag Positives** to export your labeled samples

### Workflow Tips

- Start with clear examples: Label the most obvious positive and negative samples first
- Use **View Predictions** to find high-confidence predictions to verify
- The **GraphLabelPropagationModel** works well when you have few labels but want to leverage the structure of your data
- Use **Model Hyperparameters** to tune model behavior (e.g., `{"C": 0.1}` for LinearSVM)

## Configuration

### Embedding Field

The plugin works with any embedding field in your dataset. If the selected field doesn't exist, it will compute ResNet-18 embeddings automatically.

### Model Hyperparameters

Model hyperparameters are configured directly in the panel UI for the selected model:

- **LinearSVMModel**: `C`, `max_iter`
- **RocchioPrototypeModel**: `mode`, `beta`, `gamma`, `temperature`
- **NCAMetricLearningModel**: `n_components`, `max_iter`
- **LMNNMetricLearningModel**: `n_components`, `k`, `max_iter`, `learning_rate`
- **GraphLabelPropagationModel**: `n_neighbors`, `alpha`, `max_iter`

### Advanced Settings

- **Batch Size**: Number of samples per inference batch (default: 1024)
- **Num Workers**: DataLoader workers for parallel loading (default: 0)
- **Vectorize**: Use vectorized field extraction (default: true)
- **Skip Failures**: Skip samples that fail to load (default: true)
