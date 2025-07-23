# ===== Standard Library =====
import os
import gc
import random
import logging

# ===== Data Processing =====
import numpy as np

# ===== PyTorch =====
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import torch.optim.lr_scheduler
from tqdm import tqdm

# ===== Visualization =====
import seaborn as sns
import matplotlib.pyplot as plt

# ===== Metrics =====
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix
)

import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter


# 1. Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Tensor of shape [num_classes] or float
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = torch.full_like(targets, self.alpha, dtype=torch.float).to(inputs.device)
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                raise TypeError('alpha must be float or torch.Tensor')
            ce_loss = ce_loss * alpha_t

        loss = ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()


# 4. Utility Functions
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def print_config(config, trial=None):
    logger = logging.getLogger(__name__)
    trial_id = getattr(trial, 'number', 'Manual')
    config_str = ' | '.join([f"{k}={v}" for k, v in config.items()])
    logger.info(f"[Trial {trial_id}] {config_str}")


# 5. Training and Validation

def plot_confusion_matrix_double(y_true, y_pred, class_names, dataset_type=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix - Absolute Values')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_normalized * 100, annot=True, fmt='.0f', cmap='Purples', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix - %')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    title = 'Confusion Matrix Comparison'
    if dataset_type:
        title = f'{title} {dataset_type}'

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()

def early_stopping(no_improve_epochs, patience):
    return no_improve_epochs >= patience

def clean_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
    

def plot_label_distribution_from_pt(pt_dir="Data", class_order=None, save_path=None):
    """
    Load train/val/test .pt files and plot class distribution.

    Args:
        pt_dir (str): Folder path containing train.pt, val.pt, test.pt
        class_order (list[str] or list[int], optional): Desired class order for x-axis
        save_path (str, optional): If provided, saves the plot to this path
    """

    files = {
        'Train': os.path.join(pt_dir, "train.pt"),
        'Val': os.path.join(pt_dir, "val.pt"),
        'Test': os.path.join(pt_dir, "test.pt"),
    }

    def load_label_counts(path):
        data = torch.load(path)
        labels = data['labels']
        return Counter(labels)

    # Load counts
    label_counts = {split: load_label_counts(path) for split, path in files.items()}

    # Get full class list
    if class_order:
        all_classes = class_order
    else:
        all_classes = sorted(set(label for counts in label_counts.values() for label in counts))

    # Convert to DataFrame
    df = pd.DataFrame({
        split: [label_counts[split].get(cls, 0) for cls in all_classes]
        for split in files
    }, index=all_classes)

    # Plot
    df.plot(kind='bar', figsize=(10, 6), rot=45)
    plt.title("Sample Count per Class in Train/Val/Test")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()

    return df

def plot_flux_examples_three_panels(
    pt_dir="Data",
    interp_len=4096,
    wavelength_range=(3850, 8500),
    save_path=None,
    seed=42
):
    """
    Plot one flux sample from train.pt, val.pt, test.pt into three vertically stacked subplots.

    Args:
        pt_dir (str): Directory containing .pt files
        interp_len (int): Interpolation length (default: 4096)
        wavelength_range (tuple): (wl_min, wl_max)
        save_path (str, optional): Path to save the plot image
        seed (int): Random seed for reproducibility
    """

    random.seed(seed)
    files = {
        'Train': os.path.join(pt_dir, "train.pt"),
        'Val': os.path.join(pt_dir, "val.pt"),
        'Test': os.path.join(pt_dir, "test.pt"),
    }

    wl_min, wl_max = wavelength_range
    wavelength = np.linspace(wl_min, wl_max, interp_len)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Spectral Flux Example from Train / Val / Test", fontsize=16)

    for ax, (split, path) in zip(axes, files.items()):
        data = torch.load(path)
        flux = data['flux']
        idx = random.randint(0, len(flux) - 1)
        sample_flux = flux[idx].squeeze().numpy()
        ax.plot(wavelength, sample_flux, label=f"{split} sample #{idx}")
        ax.set_ylabel("Normalized Flux")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[-1].set_xlabel("Wavelength (Å)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved flux panel to: {save_path}")
    else:
        plt.show()
