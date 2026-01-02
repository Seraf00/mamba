"""
Visualization utilities for cardiac segmentation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path


# CAMUS color scheme
CLASS_COLORS = {
    0: [0, 0, 0],        # Background - Black
    1: [255, 0, 0],      # LV Endo - Red
    2: [0, 255, 0],      # LV Epi - Green
    3: [0, 0, 255]       # LA - Blue
}

CLASS_NAMES = {
    0: 'Background',
    1: 'LV Endocardium',
    2: 'LV Epicardium',
    3: 'Left Atrium'
}


def get_colormap(num_classes: int = 4) -> ListedColormap:
    """Get colormap for segmentation visualization."""
    colors = [np.array(CLASS_COLORS.get(i, [128, 128, 128])) / 255 
              for i in range(num_classes)]
    return ListedColormap(colors)


def plot_segmentation(
    image: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    prediction: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    alpha: float = 0.5
) -> plt.Figure:
    """
    Plot segmentation results.
    
    Args:
        image: Input image (H, W) or (C, H, W)
        mask: Ground truth mask (H, W)
        prediction: Model prediction (H, W) or (C, H, W)
        title: Plot title
        save_path: Path to save figure
        show: Display figure
        alpha: Overlay transparency
        
    Returns:
        Matplotlib figure
    """
    # Convert tensors
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if prediction is not None and isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Handle channel dimension
    if image.ndim == 3:
        image = image[0] if image.shape[0] == 1 else image.transpose(1, 2, 0)
    if prediction is not None and prediction.ndim == 3:
        prediction = prediction.argmax(axis=0)
    
    # Create figure
    n_cols = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    cmap = get_colormap()
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(mask, cmap=cmap, alpha=alpha, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction overlay
    if prediction is not None:
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(prediction, cmap=cmap, alpha=alpha, vmin=0, vmax=3)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_dice', 'lr'
        save_path: Path to save figure
        show: Display figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss curves
    if 'train_loss' in history:
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
    if 'val_loss' in history:
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice curve
    if 'val_dice' in history:
        axes[0, 1].plot(epochs, history['val_dice'], 'g-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].set_title('Validation Dice Score')
        axes[0, 1].grid(True)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], 'm-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Best metrics
    if 'val_dice' in history:
        best_dice = max(history['val_dice'])
        best_epoch = history['val_dice'].index(best_dice) + 1
        
        text = f"Best Dice: {best_dice:.4f} (Epoch {best_epoch})"
        if 'val_loss' in history:
            best_loss = min(history['val_loss'])
            text += f"\nBest Loss: {best_loss:.4f}"
        
        axes[1, 1].text(0.5, 0.5, text, fontsize=14, ha='center', va='center',
                        transform=axes[1, 1].transAxes)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Best Metrics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create RGB overlay of segmentation on image.
    
    Args:
        image: Grayscale image (H, W)
        mask: Segmentation mask (H, W)
        alpha: Transparency
        
    Returns:
        RGB overlay image (H, W, 3)
    """
    # Normalize image
    if image.max() > 1:
        image = image / 255.0
    
    # Create RGB image
    rgb = np.stack([image] * 3, axis=-1)
    
    # Create colored mask
    H, W = mask.shape
    colored_mask = np.zeros((H, W, 3))
    
    for class_idx, color in CLASS_COLORS.items():
        class_mask = mask == class_idx
        colored_mask[class_mask] = np.array(color) / 255.0
    
    # Blend
    overlay = rgb.copy()
    foreground = mask > 0
    overlay[foreground] = (1 - alpha) * rgb[foreground] + alpha * colored_mask[foreground]
    
    return (overlay * 255).astype(np.uint8)


def save_predictions(
    images: torch.Tensor,
    masks: torch.Tensor,
    predictions: torch.Tensor,
    output_dir: str,
    prefix: str = 'pred'
):
    """
    Save batch of predictions as images.
    
    Args:
        images: Input images (B, C, H, W)
        masks: Ground truth (B, H, W)
        predictions: Model predictions (B, C, H, W) or (B, H, W)
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    if predictions.ndim == 4:
        predictions = predictions.argmax(axis=1)
    
    for i in range(len(images)):
        img = images[i, 0] if images[i].shape[0] == 1 else images[i].transpose(1, 2, 0)
        
        fig = plot_segmentation(
            img, masks[i], predictions[i],
            title=f'Sample {i + 1}',
            save_path=output_dir / f'{prefix}_{i:04d}.png',
            show=False
        )
        plt.close(fig)


def plot_class_distribution(
    masks: torch.Tensor,
    title: str = 'Class Distribution',
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot class distribution in dataset."""
    masks = masks.cpu().numpy().flatten()
    
    unique, counts = np.unique(masks, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = [np.array(CLASS_COLORS.get(int(c), [128, 128, 128])) / 255 
              for c in unique]
    labels = [CLASS_NAMES.get(int(c), f'Class {c}') for c in unique]
    
    bars = ax.bar(labels, counts, color=colors, edgecolor='black')
    
    ax.set_ylabel('Pixel Count')
    ax.set_title(title)
    
    # Add percentage labels
    total = counts.sum()
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_comparison_grid(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    image: np.ndarray,
    ground_truth: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create comparison grid of multiple models.
    
    Args:
        results: Dict of model_name -> (prediction, dice_score)
        image: Input image
        ground_truth: Ground truth mask
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_models = len(results)
    n_cols = min(4, n_models + 2)  # +2 for image and GT
    n_rows = (n_models + 2 + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    cmap = get_colormap()
    
    # Input image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(image, cmap='gray')
    axes[1].imshow(ground_truth, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Model predictions
    for idx, (name, (pred, dice)) in enumerate(results.items()):
        ax = axes[idx + 2]
        ax.imshow(image, cmap='gray')
        ax.imshow(pred, cmap=cmap, alpha=0.5, vmin=0, vmax=3)
        ax.set_title(f'{name}\nDice: {dice:.3f}')
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(results) + 2, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


if __name__ == '__main__':
    # Test visualization
    image = np.random.rand(256, 256)
    mask = np.random.randint(0, 4, (256, 256))
    
    plot_segmentation(image, mask, mask, title='Test', show=False)
    print("Visualization module loaded successfully")
