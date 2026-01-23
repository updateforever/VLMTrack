"""
Speed Analysis Module for Tracking Results
===========================================
Analyzes the runtime performance (FPS) of trackers based on saved time.txt files.

Usage:
    from lib.test.analysis.speed_analysis import print_speed_results, extract_speed_results
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import torch

env_path = os.path.join(os.path.dirname(__file__), '../../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation.environment import env_settings


def load_time_file(file_path):
    """
    Load timing data from a time.txt file.
    
    Args:
        file_path: Path to the time.txt file
        
    Returns:
        numpy array of frame times, or None if file doesn't exist
    """
    if not os.path.isfile(file_path):
        return None
    
    try:
        times = np.loadtxt(file_path, dtype=np.float64)
        # Handle single value case
        if times.ndim == 0:
            times = np.array([times])
        return times
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def extract_speed_results(trackers, dataset, skip_missing_seq=False, exclude_first_frame=True):
    """
    Extract speed/timing results for given trackers on a dataset.
    
    Args:
        trackers: List of Tracker instances
        dataset: List of Sequence instances
        skip_missing_seq: If True, skip sequences with missing time files
        exclude_first_frame: If True, exclude first frame from timing (initialization)
        
    Returns:
        dict containing:
            - 'sequences': list of sequence names
            - 'trackers': list of tracker info dicts
            - 'valid_sequence': list of bools indicating valid sequences
            - 'frame_times': [num_seqs, num_trackers] list of frame time arrays
            - 'fps_per_seq': [num_seqs, num_trackers] tensor of FPS values
            - 'total_frames': [num_seqs, num_trackers] tensor of frame counts
    """
    num_seqs = len(dataset)
    num_trackers = len(trackers)
    
    # Storage for results
    frame_times = [[None for _ in range(num_trackers)] for _ in range(num_seqs)]
    fps_per_seq = torch.zeros((num_seqs, num_trackers), dtype=torch.float64)
    total_frames = torch.zeros((num_seqs, num_trackers), dtype=torch.int64)
    valid_sequence = torch.ones(num_seqs, dtype=torch.uint8)
    
    for seq_id, seq in enumerate(tqdm(dataset, desc="Extracting speed results")):
        for trk_id, trk in enumerate(trackers):
            # Construct time file path (same logic as running.py)
            if seq.dataset in ['trackingnet', 'lasot', 'got10k', 'lasot_extension_subset', 
                               'otb', 'uav', 'nfs', 'tnl2k', 'soibench']:
                base_results_path = os.path.join(trk.results_dir, seq.dataset, seq.name)
            else:
                base_results_path = os.path.join(trk.results_dir, seq.name)
            
            time_file = '{}_time.txt'.format(base_results_path)
            
            times = load_time_file(time_file)
            
            if times is None:
                if skip_missing_seq:
                    valid_sequence[seq_id] = 0
                    break
                else:
                    # Try alternative path without dataset subdirectory
                    alt_time_file = os.path.join(trk.results_dir, seq.name + '_time.txt')
                    times = load_time_file(alt_time_file)
                    
                    if times is None:
                        print(f"Warning: Time file not found for {trk.display_name or trk.name} on {seq.name}")
                        valid_sequence[seq_id] = 0
                        continue
            
            # Exclude first frame if requested (first frame is initialization)
            if exclude_first_frame and len(times) > 1:
                times_for_fps = times[1:]
            else:
                times_for_fps = times
            
            frame_times[seq_id][trk_id] = times
            total_frames[seq_id, trk_id] = len(times_for_fps)
            
            # Calculate FPS
            total_time = np.sum(times_for_fps)
            if total_time > 0:
                fps_per_seq[seq_id, trk_id] = len(times_for_fps) / total_time
            else:
                fps_per_seq[seq_id, trk_id] = 0
    
    # Prepare output
    seq_names = [s.name for s in dataset]
    tracker_names = [{'name': t.name, 'param': t.parameter_name, 'run_id': t.run_id, 
                      'disp_name': t.display_name} for t in trackers]
    
    speed_data = {
        'sequences': seq_names,
        'trackers': tracker_names,
        'valid_sequence': valid_sequence.tolist(),
        'frame_times': frame_times,
        'fps_per_seq': fps_per_seq,
        'total_frames': total_frames,
    }
    
    return speed_data


def compute_speed_statistics(speed_data):
    """
    Compute various speed statistics from extracted speed data.
    
    Args:
        speed_data: Output from extract_speed_results
        
    Returns:
        dict containing speed statistics per tracker
    """
    valid_sequence = torch.tensor(speed_data['valid_sequence'], dtype=torch.bool)
    fps_per_seq = speed_data['fps_per_seq']
    total_frames = speed_data['total_frames']
    frame_times = speed_data['frame_times']
    trackers = speed_data['trackers']
    
    num_trackers = len(trackers)
    stats = {}
    
    for trk_id, trk in enumerate(trackers):
        trk_name = trk['disp_name'] or f"{trk['name']}_{trk['param']}"
        
        # Get valid sequences for this tracker
        valid_fps = fps_per_seq[valid_sequence, trk_id]
        valid_fps = valid_fps[valid_fps > 0]  # Exclude zero FPS entries
        
        if len(valid_fps) == 0:
            stats[trk_name] = {
                'mean_fps': 0.0,
                'std_fps': 0.0,
                'min_fps': 0.0,
                'max_fps': 0.0,
                'median_fps': 0.0,
                'total_frames': 0,
                'total_time': 0.0,
                'overall_fps': 0.0,
                'num_sequences': 0,
            }
            continue
        
        # Collect all frame times for this tracker
        all_times = []
        total_time = 0.0
        total_frame_count = 0
        for seq_id in range(len(speed_data['sequences'])):
            if valid_sequence[seq_id] and frame_times[seq_id][trk_id] is not None:
                times = frame_times[seq_id][trk_id]
                # Exclude first frame
                if len(times) > 1:
                    times = times[1:]
                all_times.extend(times.tolist())
                total_time += np.sum(times)
                total_frame_count += len(times)
        
        stats[trk_name] = {
            'mean_fps': valid_fps.mean().item(),
            'std_fps': valid_fps.std().item() if len(valid_fps) > 1 else 0.0,
            'min_fps': valid_fps.min().item(),
            'max_fps': valid_fps.max().item(),
            'median_fps': valid_fps.median().item(),
            'total_frames': total_frame_count,
            'total_time': total_time,
            'overall_fps': total_frame_count / total_time if total_time > 0 else 0.0,
            'num_sequences': len(valid_fps),
            'mean_frame_time_ms': np.mean(all_times) * 1000 if all_times else 0.0,
            'std_frame_time_ms': np.std(all_times) * 1000 if all_times else 0.0,
        }
    
    return stats


def generate_speed_report(stats, table_name='Speed Analysis'):
    """
    Generate a formatted speed report table.
    
    Args:
        stats: Output from compute_speed_statistics
        table_name: Title for the table
        
    Returns:
        Formatted string report
    """
    if not stats:
        return "No speed data available."
    
    # Header
    tracker_names = list(stats.keys())
    name_width = max([len(n) for n in tracker_names] + [len(table_name)]) + 5
    
    columns = ['Avg FPS', 'Std FPS', 'Min FPS', 'Max FPS', 'Med FPS', 'Overall FPS', 'Avg Time(ms)', '#Frames', '#Seqs']
    col_widths = [max(12, len(c) + 3) for c in columns]
    
    # Build header
    report = '\n' + '=' * (name_width + sum(col_widths) + len(columns) * 3 + 10) + '\n'
    report += f'{table_name}\n'
    report += '=' * (name_width + sum(col_widths) + len(columns) * 3 + 10) + '\n'
    
    report += f'{"Tracker": <{name_width}} |'
    for col, w in zip(columns, col_widths):
        report += f' {col: <{w}} |'
    report += '\n'
    report += '-' * (name_width + sum(col_widths) + len(columns) * 3 + 10) + '\n'
    
    # Data rows
    for trk_name in tracker_names:
        s = stats[trk_name]
        report += f'{trk_name: <{name_width}} |'
        report += f' {s["mean_fps"]: <{col_widths[0]}.2f} |'
        report += f' {s["std_fps"]: <{col_widths[1]}.2f} |'
        report += f' {s["min_fps"]: <{col_widths[2]}.2f} |'
        report += f' {s["max_fps"]: <{col_widths[3]}.2f} |'
        report += f' {s["median_fps"]: <{col_widths[4]}.2f} |'
        report += f' {s["overall_fps"]: <{col_widths[5]}.2f} |'
        report += f' {s["mean_frame_time_ms"]: <{col_widths[6]}.2f} |'
        report += f' {s["total_frames"]: <{col_widths[7]}} |'
        report += f' {s["num_sequences"]: <{col_widths[8]}} |'
        report += '\n'
    
    report += '=' * (name_width + sum(col_widths) + len(columns) * 3 + 10) + '\n'
    
    # Notes
    report += '\nNotes:\n'
    report += '  - Avg FPS: Average FPS across sequences (sequence-level mean)\n'
    report += '  - Overall FPS: Total frames / Total time (frame-level mean)\n'
    report += '  - First frame (initialization) is excluded from timing statistics\n'
    
    return report


def print_speed_results(trackers, dataset, report_name=None, skip_missing_seq=True, 
                        exclude_first_frame=True, return_stats=False):
    """
    Print speed analysis results for given trackers on a dataset.
    
    Args:
        trackers: List of Tracker instances
        dataset: List of Sequence instances
        report_name: Optional name for the report
        skip_missing_seq: If True, skip sequences with missing time files
        exclude_first_frame: If True, exclude first frame from timing
        return_stats: If True, also return the statistics dict
        
    Returns:
        report_text (str), and optionally stats (dict)
    """
    if report_name is None:
        report_name = 'Speed Analysis'
    
    print(f'\nExtracting speed data for {len(trackers)} trackers on {len(dataset)} sequences...')
    
    # Extract speed results
    speed_data = extract_speed_results(trackers, dataset, skip_missing_seq, exclude_first_frame)
    
    # Compute statistics
    stats = compute_speed_statistics(speed_data)
    
    # Generate report
    valid_count = sum(speed_data['valid_sequence'])
    total_count = len(speed_data['valid_sequence'])
    print(f'\nComputed speed results over {valid_count} / {total_count} sequences')
    
    report_text = generate_speed_report(stats, table_name=report_name)
    print(report_text)
    
    if return_stats:
        return report_text, stats
    return report_text


def print_speed_comparison(trackers, dataset, report_name='Speed Comparison'):
    """
    Print a simplified speed comparison table focusing on key metrics.
    
    Args:
        trackers: List of Tracker instances
        dataset: List of Sequence instances
        report_name: Name for the report
        
    Returns:
        report_text (str)
    """
    speed_data = extract_speed_results(trackers, dataset, skip_missing_seq=True)
    stats = compute_speed_statistics(speed_data)
    
    # Simplified table
    tracker_names = list(stats.keys())
    name_width = max([len(n) for n in tracker_names] + [15]) + 3
    
    report = f'\n{"=" * 60}\n{report_name}\n{"=" * 60}\n'
    report += f'{"Tracker": <{name_width}} | {"FPS (Avg±Std)": <20} | {"Overall FPS": <15}\n'
    report += f'{"-" * 60}\n'
    
    for trk_name in tracker_names:
        s = stats[trk_name]
        fps_str = f'{s["mean_fps"]:.1f} ± {s["std_fps"]:.1f}'
        report += f'{trk_name: <{name_width}} | {fps_str: <20} | {s["overall_fps"]:<15.1f}\n'
    
    report += f'{"=" * 60}\n'
    print(report)
    return report


def get_per_sequence_fps(trackers, dataset, skip_missing_seq=True):
    """
    Get per-sequence FPS for detailed analysis.
    
    Args:
        trackers: List of Tracker instances
        dataset: List of Sequence instances
        skip_missing_seq: If True, skip sequences with missing time files
        
    Returns:
        pandas-style dict with sequence names as index and tracker FPS as columns
    """
    speed_data = extract_speed_results(trackers, dataset, skip_missing_seq)
    
    valid_sequence = torch.tensor(speed_data['valid_sequence'], dtype=torch.bool)
    fps_per_seq = speed_data['fps_per_seq']
    seq_names = speed_data['sequences']
    trackers_info = speed_data['trackers']
    
    result = {'sequence': []}
    for trk in trackers_info:
        trk_name = trk['disp_name'] or f"{trk['name']}_{trk['param']}"
        result[trk_name] = []
    
    for seq_id, (seq_name, valid) in enumerate(zip(seq_names, valid_sequence.tolist())):
        if valid:
            result['sequence'].append(seq_name)
            for trk_id, trk in enumerate(trackers_info):
                trk_name = trk['disp_name'] or f"{trk['name']}_{trk['param']}"
                result[trk_name].append(fps_per_seq[seq_id, trk_id].item())
    
    return result


# ============================================================================
# Utility functions for integration with existing analysis
# ============================================================================

def add_speed_to_results(trackers, dataset, report_name, existing_eval_data=None):
    """
    Add speed statistics to existing evaluation data.
    
    Args:
        trackers: List of Tracker instances
        dataset: List of Sequence instances
        report_name: Report name for saving
        existing_eval_data: Optional existing eval_data dict to extend
        
    Returns:
        Extended eval_data dict with speed information
    """
    speed_data = extract_speed_results(trackers, dataset, skip_missing_seq=True)
    stats = compute_speed_statistics(speed_data)
    
    if existing_eval_data is not None:
        existing_eval_data['speed_stats'] = stats
        existing_eval_data['fps_per_seq'] = speed_data['fps_per_seq'].tolist()
        return existing_eval_data
    
    return {
        'speed_stats': stats,
        'fps_per_seq': speed_data['fps_per_seq'].tolist(),
        'sequences': speed_data['sequences'],
        'trackers': speed_data['trackers'],
    }
