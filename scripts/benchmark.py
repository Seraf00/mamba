"""
Benchmark script for comparing model efficiency.

Compares parameters, FLOPs, inference time, and memory usage.

Usage:
    python scripts/benchmark.py --models mamba_unet_v1 mamba_unet_v2 unet
    python scripts/benchmark.py --all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from models import get_model, MODEL_REGISTRY
from metrics import EfficiencyBenchmark, PortabilityAnalysis
from utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark model efficiency')
    
    parser.add_argument('--models', nargs='+', type=str,
                        help='Models to benchmark')
    parser.add_argument('--all', action='store_true',
                        help='Benchmark all available models')
    parser.add_argument('--mamba_types', nargs='+', type=str,
                        default=['mamba'],
                        help='Mamba types to test')
    
    parser.add_argument('--input_size', nargs=4, type=int,
                        default=[1, 1, 256, 256],
                        help='Input size (B, C, H, W)')
    parser.add_argument('--num_iterations', type=int, default=100,
                        help='Number of timing iterations')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--output', type=str, default='./benchmark_results.csv',
                        help='Output CSV file')
    parser.add_argument('--portability', action='store_true',
                        help='Run portability analysis')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    set_seed(42)
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Get models to benchmark
    if args.all:
        model_names = list(MODEL_REGISTRY.keys())
    elif args.models:
        model_names = args.models
    else:
        # Default set
        model_names = [
            'unet',
            'mamba_unet_v1',
            'mamba_unet_v2',
            'pure_mamba_unet'
        ]
    
    print(f"Benchmarking {len(model_names)} models")
    print(f"Input size: {args.input_size}")
    print(f"Device: {device}")
    print(f"Iterations: {args.num_iterations}")
    print()
    
    # Create benchmark
    benchmark = EfficiencyBenchmark(
        input_size=tuple(args.input_size),
        device=device,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup
    )
    
    # Benchmark each model
    all_results = []
    
    for model_name in model_names:
        for mamba_type in args.mamba_types:
            try:
                # Create model
                model = get_model(
                    model_name,
                    in_channels=args.input_size[1],
                    num_classes=4,
                    mamba_type=mamba_type
                )
                
                name = f"{model_name}_{mamba_type}" if 'mamba' in model_name.lower() else model_name
                
                # Run benchmark
                result = benchmark.benchmark(model, name)
                all_results.append(result)
                
                print(f"✓ {name}")
                print(f"  Params: {result.num_parameters / 1e6:.2f}M")
                print(f"  Time: {result.inference_time_ms:.2f}ms")
                print(f"  Memory: {result.memory_mb:.1f}MB")
                print()
                
                # Clean up
                del model
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"✗ {model_name}: {e}")
                continue
    
    # Print comparison
    if all_results:
        print("\n" + "=" * 80)
        benchmark.print_comparison(all_results)
        
        # Export to CSV
        benchmark.export_to_csv(all_results, args.output)
        
        # Portability analysis
        if args.portability:
            print("\nPortability Analysis (Mobile Target):")
            print("-" * 50)
            
            analyzer = PortabilityAnalysis(target_device='mobile')
            
            for result in all_results:
                analysis = analyzer.analyze(result)
                status = "✓" if analysis['portable_ready'] else "✗"
                print(f"{status} {result.model_name}")


if __name__ == '__main__':
    main()
