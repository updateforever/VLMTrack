import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                num_gpus=8, run_tag=None, debug_frames=None, disable_keyframe=False):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset.
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    if sequence is not None:
        dataset = [dataset[sequence]]

    trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
    for tracker in trackers:
        tracker.debug_frames = debug_frames
        tracker.force_use_keyframe = False if disable_keyframe else None

    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus, run_tag=run_tag)


def main():
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of config file.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default='lasot', help='Name of dataset (otb, nfs, uav, got10k_test, '
                                                                          'lasot, trackingnet, lasot_extension_subset, tnl2k,'
                                                                          'lasot_lang, otb99_lang).')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--run_tag', type=str, default=None, help='Override saved dataset folder name for this run.')
    parser.add_argument('--debug_frames', type=int, default=None, help='When debug > 0, only run the first N frames (including the init frame).')
    parser.add_argument('--disable_keyframe', action='store_true', help='Disable sparse keyframe mode and run VLM on every frame.')

    args = parser.parse_args()

    seq_name = args.sequence
    if isinstance(args.sequence, str) and args.sequence.isdigit():
        # Preserve zero-padded sequence ids like "005"; only cast plain indices such as "5".
        if not (len(args.sequence) > 1 and args.sequence.startswith('0')):
            seq_name = int(args.sequence)

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, num_gpus=args.num_gpus, run_tag=args.run_tag, debug_frames=args.debug_frames,
                disable_keyframe=args.disable_keyframe)


if __name__ == '__main__':
    main()
