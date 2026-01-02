#!/usr/bin/env python
"""
Explainability script for cardiac segmentation models.

Generate visual explanations and clinical reports for model predictions.

Usage:
    python scripts/explain.py --checkpoint path/to/model.pth --image path/to/image.npy
    python scripts/explain.py --checkpoint path/to/model.pth --data_dir path/to/data --output results/
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate explainability visualizations for cardiac segmentation models'
    )
    
    # Input
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--image', type=str, default=None,
        help='Path to single image (numpy or nifti)'
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help='Directory containing test images'
    )
    
    # Model
    parser.add_argument(
        '--model', type=str, default=None,
        help='Model name (if not stored in checkpoint)'
    )
    
    # Output
    parser.add_argument(
        '--output', type=str, default='results/explainability',
        help='Output directory for visualizations'
    )
    
    # Methods
    parser.add_argument(
        '--methods', type=str, nargs='+',
        default=['gradcam', 'attention', 'uncertainty'],
        choices=['gradcam', 'gradcam++', 'attention', 'mamba_states', 
                 'uncertainty', 'feature_maps', 'all'],
        help='Explainability methods to run'
    )
    parser.add_argument(
        '--target_class', type=int, default=1,
        help='Target class for Grad-CAM (1=LV, 2=MYO, 3=LA)'
    )
    
    # Uncertainty
    parser.add_argument(
        '--n_samples', type=int, default=10,
        help='Number of MC Dropout samples for uncertainty'
    )
    
    # Clinical report
    parser.add_argument(
        '--clinical_report', action='store_true',
        help='Generate clinical report'
    )
    
    # Device
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Device for inference'
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, model_name: Optional[str], device: str):
    """Load model from checkpoint."""
    from models import get_model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get model name from checkpoint
    if model_name is None:
        config = checkpoint.get('config', {})
        model_name = config.get('model', 'mamba_unet_v1')
    
    # Create model
    model = get_model(model_name, in_channels=1, num_classes=4)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def load_image(image_path: str, device: str) -> torch.Tensor:
    """Load and preprocess image."""
    import numpy as np
    
    path = Path(image_path)
    
    if path.suffix in ['.npy', '.npz']:
        image = np.load(image_path)
        if isinstance(image, np.lib.npyio.NpzFile):
            image = image['arr_0']
    elif path.suffix in ['.nii', '.gz']:
        import nibabel as nib
        image = nib.load(image_path).get_fdata()
    else:
        from PIL import Image
        image = np.array(Image.open(image_path).convert('L'))
    
    # Normalize
    image = image.astype(np.float32)
    image = (image - image.mean()) / (image.std() + 1e-8)
    
    # To tensor
    if image.ndim == 2:
        image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = torch.from_numpy(image).unsqueeze(0)
    
    return image.to(device)


def run_gradcam(
    model,
    image: torch.Tensor,
    target_class: int,
    output_dir: Path,
    variant: str = 'gradcam'
):
    """Run Grad-CAM visualization."""
    from explainability import GradCAM
    from utils.visualization import save_overlay
    
    print(f"Running {variant.upper()}...")
    
    # Find target layer (usually last encoder layer)
    target_layer = None
    for name, module in model.named_modules():
        if 'encoder' in name.lower() and hasattr(module, 'weight'):
            target_layer = module
    
    if target_layer is None:
        # Fallback to last conv layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
    
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image, target_class=target_class)
    
    # Save
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_np = image.squeeze().cpu().numpy()
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title(f'{variant.upper()} (Class {target_class})')
    axes[1].axis('off')
    
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(cam, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{variant}_class{target_class}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_dir / f'{variant}_class{target_class}.png'}")


def run_attention_visualization(
    model,
    image: torch.Tensor,
    output_dir: Path
):
    """Visualize attention maps."""
    from explainability import AttentionExtractor
    
    print("Extracting attention maps...")
    
    extractor = AttentionExtractor(model)
    attentions = extractor.extract(image)
    
    if not attentions:
        print("  No attention maps found in model")
        return
    
    import matplotlib.pyplot as plt
    
    n_maps = min(len(attentions), 8)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (name, attn) in enumerate(list(attentions.items())[:n_maps]):
        if attn.dim() > 2:
            attn = attn.mean(dim=tuple(range(attn.dim() - 2)))
        
        axes[i].imshow(attn.cpu().numpy(), cmap='viridis')
        axes[i].set_title(name[:20], fontsize=8)
        axes[i].axis('off')
    
    for i in range(n_maps, 8):
        axes[i].axis('off')
    
    plt.suptitle('Attention Maps')
    plt.tight_layout()
    plt.savefig(output_dir / 'attention_maps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_dir / 'attention_maps.png'}")


def run_mamba_state_visualization(
    model,
    image: torch.Tensor,
    output_dir: Path
):
    """Visualize Mamba state dynamics."""
    from explainability import MambaStateVisualizer
    
    print("Visualizing Mamba states...")
    
    visualizer = MambaStateVisualizer(model)
    
    try:
        fig = visualizer.visualize_state_evolution(image)
        fig.savefig(output_dir / 'mamba_states.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved to {output_dir / 'mamba_states.png'}")
    except Exception as e:
        print(f"  Error: {str(e)}")


def run_uncertainty_estimation(
    model,
    image: torch.Tensor,
    n_samples: int,
    output_dir: Path
):
    """Run uncertainty estimation."""
    from explainability import UncertaintyEstimator
    
    print(f"Estimating uncertainty (n={n_samples})...")
    
    estimator = UncertaintyEstimator(model, n_samples=n_samples)
    pred, probs, uncertainty = estimator.predict_with_uncertainty(image)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    img_np = image.squeeze().cpu().numpy()
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred.squeeze().cpu().numpy(), cmap='viridis')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    conf = probs.max(dim=1)[0].squeeze().cpu().numpy()
    axes[2].imshow(conf, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title('Confidence')
    axes[2].axis('off')
    
    unc = uncertainty.squeeze().cpu().numpy()
    axes[3].imshow(unc, cmap='hot')
    axes[3].set_title('Uncertainty')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved to {output_dir / 'uncertainty.png'}")


def run_feature_map_visualization(
    model,
    image: torch.Tensor,
    output_dir: Path
):
    """Visualize intermediate feature maps."""
    from explainability import FeatureMapExtractor
    
    print("Extracting feature maps...")
    
    extractor = FeatureMapExtractor(model)
    features = extractor.extract(image)
    
    if not features:
        print("  No feature maps extracted")
        return
    
    import matplotlib.pyplot as plt
    
    for layer_name, feat in list(features.items())[:4]:
        if feat.dim() < 4:
            continue
        
        feat = feat.squeeze(0)
        n_channels = min(feat.shape[0], 16)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()
        
        for i in range(n_channels):
            axes[i].imshow(feat[i].cpu().numpy(), cmap='viridis')
            axes[i].axis('off')
        
        for i in range(n_channels, 16):
            axes[i].axis('off')
        
        safe_name = layer_name.replace('.', '_')[:30]
        plt.suptitle(f'Feature Maps: {layer_name}')
        plt.tight_layout()
        plt.savefig(output_dir / f'features_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved to {output_dir}")


def generate_clinical_report(
    model,
    image: torch.Tensor,
    output_dir: Path,
    args
):
    """Generate clinical explainability report."""
    from explainability import ClinicalReportGenerator
    
    print("Generating clinical report...")
    
    generator = ClinicalReportGenerator(model)
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
        if isinstance(output, dict):
            output = output['out']
        pred = output.argmax(dim=1)
    
    report = generator.generate_report(
        image=image,
        prediction=pred,
        save_dir=output_dir
    )
    
    # Save report
    with open(output_dir / 'clinical_report.txt', 'w') as f:
        f.write(report)
    
    print(f"  Saved to {output_dir / 'clinical_report.txt'}")


def main():
    args = parse_args()
    
    # Check inputs
    if args.image is None and args.data_dir is None:
        print("Error: Must provide --image or --data_dir")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, args.device)
    print("Model loaded successfully")
    
    # Get images to process
    if args.image:
        images = [args.image]
    else:
        data_dir = Path(args.data_dir)
        images = list(data_dir.glob('*.npy')) + list(data_dir.glob('*.png'))
    
    # Expand 'all' methods
    if 'all' in args.methods:
        args.methods = ['gradcam', 'gradcam++', 'attention', 'mamba_states', 
                       'uncertainty', 'feature_maps']
    
    # Process each image
    for image_path in images:
        print(f"\nProcessing: {image_path}")
        
        # Create image output directory
        image_name = Path(image_path).stem
        img_output_dir = output_dir / image_name
        img_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image
        image = load_image(str(image_path), args.device)
        
        # Run selected methods
        if 'gradcam' in args.methods:
            run_gradcam(model, image, args.target_class, img_output_dir, 'gradcam')
        
        if 'gradcam++' in args.methods:
            run_gradcam(model, image, args.target_class, img_output_dir, 'gradcam++')
        
        if 'attention' in args.methods:
            run_attention_visualization(model, image, img_output_dir)
        
        if 'mamba_states' in args.methods:
            run_mamba_state_visualization(model, image, img_output_dir)
        
        if 'uncertainty' in args.methods:
            run_uncertainty_estimation(model, image, args.n_samples, img_output_dir)
        
        if 'feature_maps' in args.methods:
            run_feature_map_visualization(model, image, img_output_dir)
        
        if args.clinical_report:
            generate_clinical_report(model, image, img_output_dir, args)
    
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
