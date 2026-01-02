"""
Efficiency metrics for model benchmarking.

Measures parameters, FLOPs, inference time, and memory usage.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import gc


@dataclass
class EfficiencyResult:
    """Efficiency benchmark result."""
    model_name: str
    num_parameters: int
    trainable_parameters: int
    flops: Optional[int]
    inference_time_ms: float
    inference_time_std_ms: float
    memory_mb: float
    throughput_fps: float
    input_size: Tuple[int, ...]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: Only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def measure_inference_time(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 1, 256, 256),
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure model inference time.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        num_iterations: Number of measurement iterations
        warmup_iterations: Warmup iterations (not counted)
        device: Device to run on
        
    Returns:
        Dictionary with timing statistics
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_size, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    
    return {
        'mean_ms': times.mean(),
        'std_ms': times.std(),
        'min_ms': times.min(),
        'max_ms': times.max(),
        'median_ms': np.median(times),
        'throughput_fps': 1000 / times.mean()
    }


def measure_memory_usage(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 1, 256, 256),
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Measure model memory usage.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to run on
        
    Returns:
        Memory usage statistics in MB
    """
    if device != 'cuda':
        return {'warning': 'Memory measurement only supported for CUDA'}
    
    model = model.to(device)
    model.eval()
    
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Baseline memory
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_allocated()
    
    # Model memory (parameters + buffers)
    model_memory = torch.cuda.memory_allocated() - baseline_memory
    
    # Forward pass memory
    dummy_input = torch.randn(*input_size, device=device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    peak_memory = torch.cuda.max_memory_allocated()
    forward_memory = peak_memory - baseline_memory
    
    # Memory in MB
    return {
        'model_memory_mb': model_memory / (1024 ** 2),
        'forward_memory_mb': forward_memory / (1024 ** 2),
        'peak_memory_mb': peak_memory / (1024 ** 2),
        'total_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2)
    }


def estimate_flops(
    model: nn.Module,
    input_size: Tuple[int, ...] = (1, 1, 256, 256),
    device: str = 'cpu'
) -> Optional[int]:
    """
    Estimate model FLOPs using fvcore if available.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device (CPU recommended for accuracy)
        
    Returns:
        Number of FLOPs or None if fvcore not available
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        model = model.to(device)
        model.eval()
        
        dummy_input = torch.randn(*input_size, device=device)
        
        flops = FlopCountAnalysis(model, dummy_input)
        return flops.total()
        
    except ImportError:
        # Try thop as alternative
        try:
            from thop import profile
            
            model = model.to(device)
            model.eval()
            
            dummy_input = torch.randn(*input_size, device=device)
            flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            return int(flops)
            
        except ImportError:
            return None


class EfficiencyBenchmark:
    """
    Comprehensive efficiency benchmarking for models.
    
    Measures parameters, FLOPs, inference time, and memory.
    """
    
    def __init__(
        self,
        input_size: Tuple[int, ...] = (1, 1, 256, 256),
        device: str = 'cuda',
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ):
        self.input_size = input_size
        self.device = device
        self.num_iterations = num_iterations
        self.warmup_iterations = warmup_iterations
    
    def benchmark(
        self,
        model: nn.Module,
        model_name: str = 'Unknown'
    ) -> EfficiencyResult:
        """
        Run full benchmark on model.
        
        Args:
            model: PyTorch model
            model_name: Name for identification
            
        Returns:
            EfficiencyResult with all metrics
        """
        # Parameter counts
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        
        # FLOPs
        flops = estimate_flops(model, self.input_size)
        
        # Inference time
        timing = measure_inference_time(
            model,
            self.input_size,
            self.num_iterations,
            self.warmup_iterations,
            self.device
        )
        
        # Memory
        if self.device == 'cuda':
            memory = measure_memory_usage(model, self.input_size, self.device)
            memory_mb = memory['peak_memory_mb']
        else:
            memory_mb = 0.0
        
        return EfficiencyResult(
            model_name=model_name,
            num_parameters=total_params,
            trainable_parameters=trainable_params,
            flops=flops,
            inference_time_ms=timing['mean_ms'],
            inference_time_std_ms=timing['std_ms'],
            memory_mb=memory_mb,
            throughput_fps=timing['throughput_fps'],
            input_size=self.input_size
        )
    
    def compare_models(
        self,
        models: Dict[str, nn.Module]
    ) -> List[EfficiencyResult]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            
        Returns:
            List of EfficiencyResult for each model
        """
        results = []
        
        for name, model in models.items():
            print(f"Benchmarking {name}...")
            result = self.benchmark(model, name)
            results.append(result)
            
            # Clear cache between models
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        
        return results
    
    def print_comparison(self, results: List[EfficiencyResult]):
        """Print formatted comparison table."""
        print("\n" + "=" * 100)
        print("EFFICIENCY COMPARISON")
        print("=" * 100)
        
        # Header
        print(f"{'Model':<25} {'Params (M)':<12} {'FLOPs (G)':<12} "
              f"{'Time (ms)':<12} {'Memory (MB)':<12} {'FPS':<10}")
        print("-" * 100)
        
        for r in results:
            params_m = r.num_parameters / 1e6
            flops_g = r.flops / 1e9 if r.flops else 0
            
            print(f"{r.model_name:<25} {params_m:<12.2f} {flops_g:<12.2f} "
                  f"{r.inference_time_ms:<12.2f} {r.memory_mb:<12.1f} "
                  f"{r.throughput_fps:<10.1f}")
        
        print("=" * 100)
        
        # Find best in each category
        if len(results) > 1:
            print("\nBest in category:")
            
            min_params = min(results, key=lambda x: x.num_parameters)
            print(f"  Smallest model: {min_params.model_name} "
                  f"({min_params.num_parameters / 1e6:.2f}M params)")
            
            min_time = min(results, key=lambda x: x.inference_time_ms)
            print(f"  Fastest inference: {min_time.model_name} "
                  f"({min_time.inference_time_ms:.2f}ms)")
            
            min_memory = min(results, key=lambda x: x.memory_mb)
            print(f"  Lowest memory: {min_memory.model_name} "
                  f"({min_memory.memory_mb:.1f}MB)")
    
    def export_to_csv(
        self,
        results: List[EfficiencyResult],
        filepath: str
    ):
        """Export results to CSV."""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Model', 'Parameters', 'Trainable Params', 'FLOPs',
                'Inference Time (ms)', 'Time Std (ms)', 'Memory (MB)',
                'Throughput (FPS)', 'Input Size'
            ])
            
            # Data
            for r in results:
                writer.writerow([
                    r.model_name,
                    r.num_parameters,
                    r.trainable_parameters,
                    r.flops or 'N/A',
                    f"{r.inference_time_ms:.2f}",
                    f"{r.inference_time_std_ms:.2f}",
                    f"{r.memory_mb:.1f}",
                    f"{r.throughput_fps:.1f}",
                    str(r.input_size)
                ])
        
        print(f"Results exported to {filepath}")


class PortabilityAnalysis:
    """
    Analyze model suitability for portable/edge deployment.
    """
    
    # Thresholds for portable deployment
    PORTABLE_THRESHOLDS = {
        'max_params_m': 5.0,  # 5M parameters
        'max_inference_ms': 50,  # 50ms (20 FPS)
        'max_memory_mb': 500,  # 500 MB
    }
    
    def __init__(self, target_device: str = 'mobile'):
        self.target_device = target_device
        
        # Adjust thresholds for target
        if target_device == 'mobile':
            self.thresholds = self.PORTABLE_THRESHOLDS.copy()
        elif target_device == 'edge':
            self.thresholds = {
                'max_params_m': 10.0,
                'max_inference_ms': 100,
                'max_memory_mb': 1000,
            }
        else:
            self.thresholds = {
                'max_params_m': 50.0,
                'max_inference_ms': 200,
                'max_memory_mb': 4000,
            }
    
    def analyze(self, result: EfficiencyResult) -> Dict[str, bool]:
        """Analyze if model meets portability requirements."""
        params_m = result.num_parameters / 1e6
        
        analysis = {
            'meets_param_requirement': params_m <= self.thresholds['max_params_m'],
            'meets_speed_requirement': result.inference_time_ms <= self.thresholds['max_inference_ms'],
            'meets_memory_requirement': result.memory_mb <= self.thresholds['max_memory_mb'],
        }
        
        analysis['portable_ready'] = all(analysis.values())
        
        return analysis
    
    def print_analysis(self, result: EfficiencyResult):
        """Print portability analysis."""
        analysis = self.analyze(result)
        
        print(f"\nPortability Analysis for {result.model_name}")
        print(f"Target: {self.target_device}")
        print("-" * 50)
        
        params_m = result.num_parameters / 1e6
        
        status = "✓" if analysis['meets_param_requirement'] else "✗"
        print(f"{status} Parameters: {params_m:.2f}M "
              f"(max: {self.thresholds['max_params_m']}M)")
        
        status = "✓" if analysis['meets_speed_requirement'] else "✗"
        print(f"{status} Inference: {result.inference_time_ms:.2f}ms "
              f"(max: {self.thresholds['max_inference_ms']}ms)")
        
        status = "✓" if analysis['meets_memory_requirement'] else "✗"
        print(f"{status} Memory: {result.memory_mb:.1f}MB "
              f"(max: {self.thresholds['max_memory_mb']}MB)")
        
        print("-" * 50)
        if analysis['portable_ready']:
            print("✓ Model is READY for portable deployment")
        else:
            print("✗ Model does NOT meet portable requirements")


if __name__ == '__main__':
    # Example usage
    import torch.nn as nn
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 4, 1)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return self.conv3(x)
    
    model = SimpleModel()
    
    benchmark = EfficiencyBenchmark(device='cpu')
    result = benchmark.benchmark(model, 'SimpleModel')
    
    print(f"Parameters: {result.num_parameters:,}")
    print(f"Inference time: {result.inference_time_ms:.2f}ms")
    print(f"Throughput: {result.throughput_fps:.1f} FPS")
    
    # Portability analysis
    analyzer = PortabilityAnalysis(target_device='mobile')
    analyzer.print_analysis(result)
