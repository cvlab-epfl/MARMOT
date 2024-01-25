import os
import sys
sys.path.append('..')

from utils.io_utils import get_landmarks, get_landmarks_global, generate_minimal_tree, get_intrinsics, get_filenames

from utils.io_utils import write_json
import argparse


def main():
    # get interval id from command line
    parser = argparse.ArgumentParser()
    # parser.add_argument('--interval', type=int, default=None)
    parser.add_argument('--opensfm_dir', type=str, default='/Users/grosche/Documents/GitHub/CVLAB/MARMOT/data/0-calibration/opensfm')
    args = parser.parse_args()
    # interval_id = args.interval

    opensfm_dir = args.opensfm_dir

    # get landmarks
    landmarks = get_landmarks(opensfm_dir)
    landmarks_global = get_landmarks_global(opensfm_dir)
    minimal_tree = generate_minimal_tree(opensfm_dir)
    intrinsics = get_intrinsics(os.path.join(opensfm_dir, 'undistorted'))
    filenames = get_filenames(os.path.join(opensfm_dir, 'undistorted'))

    # save the files
    write_json(os.path.join(opensfm_dir, 'landmarks.json'), landmarks)
    write_json(os.path.join(opensfm_dir, 'landmarks_global.json'), landmarks_global)
    write_json(os.path.join(opensfm_dir, 'setup.json'), minimal_tree)
    write_json(os.path.join(opensfm_dir, 'intrinsics.json'), intrinsics)
    write_json(os.path.join(opensfm_dir, 'filenames.json'), filenames)

if __name__ == '__main__':
    main()

