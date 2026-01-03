"""
Validate CAMUS Dataset
----------------------
This script validates the CAMUS dataset by checking for missing or corrupted files.
It reports which patients/views/phases have missing data.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.camus_dataset import CAMUSDataset
import warnings


def validate_dataset(data_dir: str, split: str = None, include_sequences: bool = False):
    """
    Validate the CAMUS dataset and report missing files.
    
    Args:
        data_dir: Path to CAMUS dataset
        split: Optional split to validate (train/val/test)
        include_sequences: Whether to validate half sequences
    """
    print(f"\n{'='*70}")
    print(f"CAMUS Dataset Validation")
    print(f"{'='*70}")
    print(f"Data directory: {data_dir}")
    print(f"Split: {split if split else 'all'}")
    print(f"Include sequences: {include_sequences}")
    print(f"{'='*70}\n")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Create dataset
        try:
            dataset = CAMUSDataset(
                root_dir=data_dir,
                split=split,
                include_sequences=include_sequences
            )
            
            # Print summary
            print(f"✓ Dataset loaded successfully")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Total patients: {len(dataset.patients)}")
            
            # Report warnings
            if len(w) > 0:
                print(f"\n{'='*70}")
                print(f"WARNINGS ({len(w)} total)")
                print(f"{'='*70}")
                
                missing_images = []
                missing_segs = []
                missing_sequences = []
                other_warnings = []
                
                for warning in w:
                    msg = str(warning.message)
                    if "Missing image:" in msg:
                        missing_images.append(msg.replace("Missing image: ", ""))
                    elif "Missing segmentation:" in msg:
                        missing_segs.append(msg.replace("Missing segmentation: ", ""))
                    elif "Missing half sequence" in msg:
                        missing_sequences.append(msg)
                    elif "Skipped" in msg and "samples" in msg:
                        print(f"\n{msg}")
                    else:
                        other_warnings.append(msg)
                
                if missing_images:
                    print(f"\nMissing Images ({len(missing_images)}):")
                    for img in sorted(set(missing_images))[:20]:  # Show first 20
                        print(f"  - {img}")
                    if len(missing_images) > 20:
                        print(f"  ... and {len(missing_images) - 20} more")
                
                if missing_segs:
                    print(f"\nMissing Segmentations ({len(missing_segs)}):")
                    for seg in sorted(set(missing_segs))[:20]:  # Show first 20
                        print(f"  - {seg}")
                    if len(missing_segs) > 20:
                        print(f"  ... and {len(missing_segs) - 20} more")
                
                if missing_sequences:
                    print(f"\nMissing Sequences ({len(missing_sequences)}):")
                    for seq in sorted(set(missing_sequences))[:10]:
                        print(f"  - {seq}")
                    if len(missing_sequences) > 10:
                        print(f"  ... and {len(missing_sequences) - 10} more")
                
                if other_warnings:
                    print(f"\nOther Warnings ({len(other_warnings)}):")
                    for warn in other_warnings[:10]:
                        print(f"  - {warn}")
                    if len(other_warnings) > 10:
                        print(f"  ... and {len(other_warnings) - 10} more")
            else:
                print("\n✓ No missing files detected!")
            
            # Sample statistics
            if len(dataset) > 0:
                print(f"\n{'='*70}")
                print(f"SAMPLE STATISTICS")
                print(f"{'='*70}")
                
                # Count by patient
                patient_counts = {}
                sequence_count = 0
                frame_count = 0
                
                for sample in dataset.samples:
                    patient_id = sample['patient_id']
                    patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
                    
                    if sample['is_sequence']:
                        sequence_count += 1
                    else:
                        frame_count += 1
                
                print(f"  ED/ES frames: {frame_count}")
                print(f"  Sequence frames: {sequence_count}")
                print(f"  Avg samples per patient: {len(dataset) / len(patient_counts):.1f}")
                
                # Check balance
                views = set(s['view'] for s in dataset.samples if not s['is_sequence'])
                phases = set(s['phase'] for s in dataset.samples if not s['is_sequence'])
                print(f"  Views: {sorted(views)}")
                print(f"  Phases: {sorted(phases)}")
            
            print(f"\n{'='*70}")
            print(f"VALIDATION COMPLETE")
            print(f"{'='*70}\n")
            
            return len(w) == 0
            
        except Exception as e:
            print(f"\n✗ ERROR loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Validate CAMUS dataset for missing or corrupted files'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to CAMUS dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        choices=['train', 'val', 'test', None],
        help='Dataset split to validate (default: all)'
    )
    parser.add_argument(
        '--include_sequences',
        action='store_true',
        help='Also validate half sequences'
    )
    
    args = parser.parse_args()
    
    success = validate_dataset(
        data_dir=args.data_dir,
        split=args.split,
        include_sequences=args.include_sequences
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
