#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import collections.abc as collections
from pathlib import Path
from typing import Dict, List, Union, Optional
import numpy as np

from hloc.utils.parsers import parse_image_lists, parse_retrieval
from hloc.utils.io import list_h5_names
from hloc import pairs_from_retrieval

from utils.log_utils import log

def main(
        output: Path,
        image_dir: Path,
        window_size: Optional[int] = 10,
        quadratic: bool = False,
        loop_closure: bool = False,
        retrieval_path: Optional[Union[Path, str]] = None,
        retrieval_interval: Optional[int] = 5,
        num_loc: Optional[int] = 5):

    view_dirs = [image_dir / d for d in os.listdir(image_dir) if os.path.isdir(image_dir / d) if '360' in d]
    view_files = {d: sorted(os.listdir(d)) for d in view_dirs}

    pairs = []
    for frame_index in range(len(view_files[view_dirs[0]]) - 1):
        for view_dir in view_dirs:
            for view_dir_2 in view_dirs:
                for offset in range(1, window_size + 1):
                    if frame_index + offset < len(view_files[view_dir]):
                        current_image = view_dir / view_files[view_dir][frame_index]
                        next_image = view_dir_2 / view_files[view_dir_2][frame_index + offset]
                        pairs.append((str(current_image).split('images/')[-1], str(next_image).split('images/')[-1]))

                    if quadratic:
                        q = 2**offset
                        if frame_index + q < len(view_files[view_dir]):
                            next_image_quad = view_dir / view_files[view_dir][frame_index + q]
                            pairs.append((str(current_image).split('images/')[-1], str(next_image_quad).split('images/')[-1]))

    if loop_closure:
        retrieval_pairs_tmp = output.parent / f'retrieval-pairs-tmp.txt'
        names_q = [str(view_dirs[0] / f) for f in view_files[view_dirs[0]]]
        query_list = names_q[::retrieval_interval]
        N = len(names_q)
        M = len(query_list)
        match_mask = np.zeros((M, N), dtype=bool)

        for i in range(M):
            for k in range(window_size + 1):
                if i * retrieval_interval - k >= 0 and i * retrieval_interval - k < N:
                    match_mask[i][i * retrieval_interval - k] = 1
                if i * retrieval_interval + k >= 0 and i * retrieval_interval + k < N:
                    match_mask[i][i * retrieval_interval + k] = 1

                if quadratic:
                    if i * retrieval_interval - 2**k >= 0 and i * retrieval_interval - 2**k < N:
                        match_mask[i][i * retrieval_interval - 2**k] = 1
                    if i * retrieval_interval + 2**k >= 0 and i * retrieval_interval + 2**k < N:
                        match_mask[i][i * retrieval_interval + 2**k] = 1

        pairs_from_retrieval.main(
            retrieval_path, retrieval_pairs_tmp, num_matched=num_loc, match_mask=match_mask, db_list=names_q, query_list=query_list)

        retrieval = parse_retrieval(retrieval_pairs_tmp)

        for key, val in retrieval.items():
            for match in val:
                pairs.append((key, match))

        os.unlink(retrieval_pairs_tmp)

    log.info(f'Found {len(pairs)} pairs.')
    log.spam(f"Pairs: {pairs}")

    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a list of image pairs based on the sequence of images from a 360 camera")
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--image_dir', required=True, type=Path,
                        help="Directory containing the cubemap views (e.g., cam360_view_forward, cam360_view_left, etc.)")
    parser.add_argument('--window_size', type=int, default=5,
                        help="Size of the window of images to match, default: %(default)s")
    parser.add_argument('--quadratic', action="store_true",
                        help="Pair elements with quadratic overlap")
    parser.add_argument('--loop_closure', action="store_true",
                        help="Do retrieval to look for possible loop closing pairs")
    parser.add_argument('--retrieval_path', type=Path,
                        help="Path to retrieval features, necessary for loop closure")
    parser.add_argument('--retrieval_interval', type=int, default=5,
                        help="Trigger retrieval every retrieval_interval frames, default: %(default)s")
    parser.add_argument('--num_loc', type=int, default=5,
                        help='Number of image pairs for loc, default: %(default)s')
    args = parser.parse_args()
    main(**args.__dict__)
