import math
import cv2
import numpy as np
from random import sample
from typing import List, Optional
from utils.multiview_utils import MultiviewVids, Camera


def get_background(cam, min_idx, max_idx, nb_frame_for_average=10):
    """
    Get the background image of a camera object by averaging a random sample
    of frames.

    Args:
        cam: Camera object to sample frames from.
        min_idx: Minimum index of frames to sample.
        max_idx: Maximum index of frames to sample.
        nb_frame_for_average: Number of frames to use for averaging.

    Returns:
        Grayscale image representing the median of the sampled frames.
    """
    frame_indices = sorted(min_idx + ((max_idx - min_idx) 
                     * np.random.uniform(size 
                                         = nb_frame_for_average)).astype(int))

    frames = cam.extract(frame_indices)
    
    median_frame = np.median(frames, axis=0).astype(np.uint8)
    median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    
    return median_frame

def contains_people_score(frame, background):
    """
    Calculate a score based on how much a frame contains people compared to a 
    background image.

    Args:
        frame: Image to score.
        background: Background image to compare against.

    Returns:
        Score representing how much the frame contains people compared to the 
        background image.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray, background)
    _, thres = cv2.threshold(frame_diff, 50, 1, cv2.THRESH_BINARY)

    score = np.count_nonzero(thres) / (thres.shape[0] * thres.shape[1])
    
    return score


def sample_frames(cam:Camera, nb_frame:int, min_idx:int=0, max_idx:int=math.inf, 
                  sample_multiplier:int=10, exclude:Optional[List]=None):
    """
    Randomly sample frames from a camera object.

    Args:
        cam: Camera object to sample frames from.
        nb_frame: Number of frames to sample.
        min_idx: Minimum index of frames to sample.
        max_idx: Maximum index of frames to sample.
        sample_multiplier: Multiplier for the number of frames to sample.
        exclude: List of frame indices to exclude from sampling.

    Returns:
        List of indices of the sampled frames.
    """
    max_idx = min(max_idx, cam.get_max_frame_id())


    if exclude is None:
        exclude = []

    # Get background by averaging
    background = get_background(cam, min_idx, max_idx)
    
    # Sample frames between min_idx and max_idx and avoid frames in exclude
    frame_indices = []
    while len(frame_indices) < sample_multiplier * nb_frame:
        frame_indices.extend((min_idx + (max_idx - min_idx)
                              * np.random.uniform(size=sample_multiplier
                                                  * nb_frame)).astype(int))
        
        frame_indices = list(set(frame_indices) - set(exclude))

    # Score frames based on how much they contain people
    frame_score = []
    # frame_indices = [frame_indices[i] + cam.first_frame 
    #                          for i in range(len(frame_indices))]
    frames = cam.extract(frame_indices)
    for frame in frames:
        frame_score.append(contains_people_score(frame, background))

    # Sort frames by score and return the top nb_frame frames
    sorted_frame_indices = [x for _, x in sorted(zip(frame_score, 
                                                     frame_indices), 
                                                     key=lambda pair: pair[0], 
                                                     reverse=True)]
    
    return sorted_frame_indices[:nb_frame]


def sample_sequence(mvv: MultiviewVids, seq_len: int, frame_interval: int = 1, 
                    min_idx: int = 0, max_idx: int = math.inf, 
                    sample_multiplier: int = 10, 
                    percentage:int = 50) -> List[int]:
    """
    Samples a sequence of frames from a multi-view video instance.

    Parameters:
    mvv (MultiViewVideo): A multi-view video object.
    seq_len (int): The length of the sequence to sample.
    frame_interval (int): The interval between frames in the sequence.
    min_idx (int): The minimum frame index to sample from.
    max_idx (int): The maximum frame index to sample from.
    sample_multiplier (int): The number of sequences to sample.
    sample_interval (int): The interval between samples in the sequence.

    Returns:
    list: A list of frame indices representing the sampled sequence.
    """
    # Reduce max_idx such that it's always possible to sample a full sequence
    max_idx = min(max_idx, mvv.get_max_frame_id())
    max_idx -= seq_len*frame_interval
    assert min_idx < max_idx, "Impossible to sample a sequence from video file"
    
    # Get background by averaging
    backgrounds = [get_background(cam, min_idx, max_idx) for cam in mvv]

    # Sample sequence start indices
    seq_starts = (min_idx + (max_idx - min_idx) 
                  * np.random.uniform(size=sample_multiplier)).astype(int)
    
    seq_score = []
    seq_idx_list = []
    for s_start in seq_starts:
        seq_idx = list(range(s_start, s_start+seq_len*frame_interval, 
                             frame_interval))
        seq_subsample = sample(seq_idx, int(len(seq_idx) * percentage / 100))

        s_frames = mvv.extract_mv(seq_subsample).values()

        s_score = sum([sum([contains_people_score(frame, background) 
                            for frame in frames]) / len(frames) 
                            for frames, background 
                            in zip(s_frames, backgrounds)]) / len(backgrounds)
        
        seq_score.append(s_score)
        seq_idx_list.append(seq_idx)
    
    # Sort sequences by score and return the first one
    sorted_frame_indices = [x for _, x in sorted(zip(seq_score, seq_idx_list), 
                                                 key=lambda pair: pair[0])]
    return sorted_frame_indices[0]  