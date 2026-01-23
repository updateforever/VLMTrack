"""
Speed Analysis Script for Tracking Results
============================================
Analyzes FPS/runtime performance based on saved time.txt files.

Usage:
    python tracking/analysis_speed.py

Example:
    Modify the trackers list and dataset_name below, then run:
    python tracking/analysis_speed.py
"""
import _init_paths
import matplotlib.pyplot as plt

from lib.test.analysis.speed_analysis import print_speed_results, print_speed_comparison, get_per_sequence_fps
from lib.test.evaluation import get_dataset, trackerlist


def main():
    # ============================================================================
    # Configuration - Modify these settings as needed
    # ============================================================================
    
    # Dataset to analyze
    dataset_name = 'lasot'
    # Options: 'lasot', 'lasot_extension_subset', 'trackingnet', 'got10k', 
    #          'otb', 'uav', 'nfs', 'tnl2k', 'soibench', etc.
    
    # Trackers to analyze - add your trackers here
    trackers = []
    
    # Example: Add trackers using trackerlist
    trackers.extend(trackerlist(
        name='sutrack', 
        parameter_name='sutrack_b224', 
        dataset_name=dataset_name,
        run_ids=None, 
        display_name='SUTrack-B224'
    ))
    
    # You can add more trackers:
    # trackers.extend(trackerlist(
    #     name='sutrack', 
    #     parameter_name='sutrack_l384', 
    #     dataset_name=dataset_name,
    #     run_ids=None, 
    #     display_name='SUTrack-L384'
    # ))
    
    # ============================================================================
    # Analysis
    # ============================================================================
    
    print(f"\n{'='*60}")
    print(f"Speed Analysis for Dataset: {dataset_name}")
    print(f"Number of Trackers: {len(trackers)}")
    print(f"{'='*60}\n")
    
    # Load dataset
    dataset = get_dataset(dataset_name)
    print(f"Loaded {len(dataset)} sequences from {dataset_name}")
    
    # Option 1: Detailed speed report
    print("\n" + "="*60)
    print("DETAILED SPEED REPORT")
    print("="*60)
    report_text, stats = print_speed_results(
        trackers, 
        dataset, 
        report_name=f'{dataset_name}_speed',
        skip_missing_seq=True,
        exclude_first_frame=True,  # First frame is initialization, usually slower
        return_stats=True
    )
    
    # Option 2: Simplified comparison (uncomment if needed)
    # print("\n" + "="*60)
    # print("SIMPLIFIED COMPARISON")
    # print("="*60)
    # print_speed_comparison(trackers, dataset, report_name=f'{dataset_name}_speed')
    
    # Option 3: Get per-sequence FPS for custom analysis (uncomment if needed)
    # per_seq_fps = get_per_sequence_fps(trackers, dataset)
    # print("\nPer-sequence FPS data available for custom analysis")
    # print(f"Columns: {list(per_seq_fps.keys())}")
    # print(f"Number of sequences: {len(per_seq_fps['sequence'])}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for trk_name, s in stats.items():
        print(f"{trk_name}:")
        print(f"  - Average FPS: {s['mean_fps']:.2f} ± {s['std_fps']:.2f}")
        print(f"  - Overall FPS: {s['overall_fps']:.2f}")
        print(f"  - Avg Frame Time: {s['mean_frame_time_ms']:.2f} ms")
        print(f"  - Total Frames: {s['total_frames']}")
        print()


if __name__ == '__main__':
    main()
