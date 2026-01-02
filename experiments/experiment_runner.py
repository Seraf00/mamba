"""
Experiment Runner for automated experiments.

Manages training and evaluation of multiple models with different configurations.
"""

import os
import yaml
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import logging
from copy import deepcopy


class ExperimentRunner:
    """
    Automated experiment runner for cardiac segmentation models.
    
    Supports:
    - Running multiple models with different configurations
    - Hyperparameter sweeps
    - Result aggregation and comparison
    - Checkpoint management
    
    Args:
        config_path: Path to experiment configuration file
        base_dir: Base directory for saving results
        device: Device for training
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        base_dir: str = 'results',
        device: str = 'cuda'
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._default_config()
        
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.results: Dict[str, Dict] = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _default_config(self) -> Dict:
        """Return default experiment configuration."""
        return {
            'experiment_name': 'cardiac_segmentation',
            'models': ['unet_v1', 'mamba_unet_v1'],
            'dataset': {
                'name': 'camus',
                'root': 'data/CAMUS',
                'img_size': 256
            },
            'training': {
                'epochs': 100,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'optimizer': 'adamw',
                'scheduler': 'cosine'
            },
            'evaluation': {
                'metrics': ['dice', 'iou', 'hd95'],
                'save_predictions': True
            }
        }
    
    def _setup_logging(self):
        """Setup logging for experiments."""
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'experiment_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_experiment(
        self,
        experiment_name: Optional[str] = None,
        models: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Dict]:
        """
        Run experiment with specified models.
        
        Args:
            experiment_name: Name for this experiment run
            models: List of model names to run
            **kwargs: Override configuration parameters
            
        Returns:
            Dictionary of results for each model
        """
        experiment_name = experiment_name or self.config.get(
            'experiment_name', 
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        models = models or self.config.get('models', ['unet_v1'])
        
        # Create experiment directory
        exp_dir = self.base_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting experiment: {experiment_name}")
        self.logger.info(f"Models to train: {models}")
        
        results = {}
        
        for model_name in models:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Training model: {model_name}")
            self.logger.info(f"{'='*50}\n")
            
            try:
                model_results = self._run_single_model(
                    model_name=model_name,
                    exp_dir=exp_dir,
                    **kwargs
                )
                results[model_name] = model_results
                
                self.logger.info(f"Model {model_name} completed successfully")
                self.logger.info(f"Results: {model_results}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Save aggregated results
        self._save_results(results, exp_dir)
        
        return results
    
    def _run_single_model(
        self,
        model_name: str,
        exp_dir: Path,
        **kwargs
    ) -> Dict:
        """Run training and evaluation for a single model."""
        from models import get_model
        from data import CAMUSDataset, get_transforms
        from training import Trainer
        from metrics import SegmentationMetrics
        from torch.utils.data import DataLoader
        
        # Merge configuration
        config = deepcopy(self.config)
        config.update(kwargs)
        
        # Create model
        model = get_model(
            model_name,
            in_channels=1,
            num_classes=4,
            **config.get('model_kwargs', {})
        )
        model = model.to(self.device)
        
        # Setup data
        dataset_config = config['dataset']
        train_transform = get_transforms('train', img_size=dataset_config.get('img_size', 256))
        val_transform = get_transforms('val', img_size=dataset_config.get('img_size', 256))
        
        train_dataset = CAMUSDataset(
            root=dataset_config['root'],
            split='train',
            transform=train_transform
        )
        val_dataset = CAMUSDataset(
            root=dataset_config['root'],
            split='val',
            transform=val_transform
        )
        
        train_config = config['training']
        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        # Setup trainer
        model_dir = exp_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={
                'epochs': train_config['epochs'],
                'lr': train_config['learning_rate'],
                'optimizer': train_config['optimizer'],
                'scheduler': train_config['scheduler'],
                'checkpoint_dir': str(model_dir),
                'device': self.device
            }
        )
        
        # Train
        train_results = trainer.train()
        
        # Evaluate
        eval_results = self._evaluate_model(model, val_loader)
        
        return {
            'training': train_results,
            'evaluation': eval_results
        }
    
    def _evaluate_model(
        self,
        model: nn.Module,
        val_loader
    ) -> Dict:
        """Evaluate model on validation set."""
        from metrics import SegmentationMetrics
        
        metrics = SegmentationMetrics(num_classes=4)
        model.eval()
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = model(images)
                if isinstance(outputs, dict):
                    outputs = outputs['out']
                
                preds = outputs.argmax(dim=1)
                metrics.update(preds, masks)
        
        return metrics.compute()
    
    def _save_results(self, results: Dict, exp_dir: Path):
        """Save experiment results."""
        # Save as JSON
        results_file = exp_dir / 'results.json'
        
        # Convert non-serializable values
        serializable = self._make_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Make object JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, 'item'):  # numpy/torch scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy/torch array
            return obj.tolist()
        else:
            return str(obj)
    
    def run_hyperparameter_sweep(
        self,
        model_name: str,
        param_grid: Dict[str, List],
        experiment_name: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Run hyperparameter sweep.
        
        Args:
            model_name: Model to tune
            param_grid: Dictionary of parameter names to list of values
            experiment_name: Name for sweep experiment
            
        Returns:
            Results for each hyperparameter combination
        """
        from itertools import product
        
        experiment_name = experiment_name or f"sweep_{model_name}"
        exp_dir = self.base_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        self.logger.info(f"Starting hyperparameter sweep with {len(combinations)} combinations")
        
        all_results = {}
        
        for i, values in enumerate(combinations):
            params = dict(zip(param_names, values))
            run_name = f"run_{i:03d}_" + "_".join(f"{k}={v}" for k, v in params.items())
            
            self.logger.info(f"\nRun {i+1}/{len(combinations)}: {params}")
            
            try:
                # Update config with current parameters
                config_override = self._params_to_config(params)
                
                results = self._run_single_model(
                    model_name=model_name,
                    exp_dir=exp_dir / run_name,
                    **config_override
                )
                
                all_results[run_name] = {
                    'params': params,
                    'results': results
                }
                
            except Exception as e:
                self.logger.error(f"Error in run {run_name}: {str(e)}")
                all_results[run_name] = {
                    'params': params,
                    'error': str(e)
                }
        
        # Find best configuration
        best_run = self._find_best_run(all_results)
        self.logger.info(f"\nBest configuration: {best_run}")
        
        # Save all results
        self._save_results(all_results, exp_dir)
        
        return all_results
    
    def _params_to_config(self, params: Dict) -> Dict:
        """Convert flat parameters to nested config."""
        config = {'training': {}, 'model_kwargs': {}}
        
        for key, value in params.items():
            if key in ['learning_rate', 'lr']:
                config['training']['learning_rate'] = value
            elif key in ['batch_size']:
                config['training']['batch_size'] = value
            elif key in ['epochs']:
                config['training']['epochs'] = value
            else:
                config['model_kwargs'][key] = value
        
        return config
    
    def _find_best_run(self, results: Dict) -> Optional[str]:
        """Find best run based on validation Dice score."""
        best_run = None
        best_dice = 0
        
        for run_name, run_data in results.items():
            if 'error' in run_data:
                continue
            
            try:
                dice = run_data['results']['evaluation'].get('dice_mean', 0)
                if dice > best_dice:
                    best_dice = dice
                    best_run = run_name
            except (KeyError, TypeError):
                continue
        
        return best_run
    
    def compare_models(
        self,
        results: Optional[Dict] = None,
        results_path: Optional[str] = None
    ) -> None:
        """
        Compare model results and generate comparison report.
        
        Args:
            results: Results dictionary
            results_path: Path to results JSON file
        """
        if results is None and results_path:
            with open(results_path, 'r') as f:
                results = json.load(f)
        
        if results is None:
            self.logger.error("No results provided for comparison")
            return
        
        # Create comparison table
        self.logger.info("\n" + "="*80)
        self.logger.info("MODEL COMPARISON")
        self.logger.info("="*80)
        
        headers = ['Model', 'Dice', 'IoU', 'HD95', 'Parameters', 'GFLOPs']
        self.logger.info(f"{headers[0]:<25} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10}")
        self.logger.info("-"*55)
        
        for model_name, model_results in results.items():
            if 'error' in model_results:
                self.logger.info(f"{model_name:<25} ERROR: {model_results['error']}")
                continue
            
            eval_results = model_results.get('evaluation', {})
            dice = eval_results.get('dice_mean', 'N/A')
            iou = eval_results.get('iou_mean', 'N/A')
            hd95 = eval_results.get('hd95_mean', 'N/A')
            
            if isinstance(dice, float):
                dice = f"{dice:.4f}"
            if isinstance(iou, float):
                iou = f"{iou:.4f}"
            if isinstance(hd95, float):
                hd95 = f"{hd95:.2f}"
            
            self.logger.info(f"{model_name:<25} {dice:<10} {iou:<10} {hd95:<10}")


if __name__ == '__main__':
    # Example usage
    runner = ExperimentRunner()
    
    # Run experiment with default models
    results = runner.run_experiment(
        experiment_name='test_experiment',
        models=['unet_v1', 'mamba_unet_v1']
    )
    
    # Compare results
    runner.compare_models(results)
