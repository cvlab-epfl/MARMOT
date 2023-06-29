"""
Implements:
    add_metadata
    get_cam_names
    get_frame_from_file
"""

import os
import cv2
import numpy as np
from utils.log_utils import log
from configs.arguments import get_config_dict
from pathlib import Path
from typing import List
import re

def get_cam_names(folder: str, omni_tag:str = '') -> List[str]:
    """
    Returns a list of all the camera directories in the folder.
    Cam directories should contain 'cam'.

    Arguments:
    ----------
    folder(str): path to the parent directory of the camera files
    """
    pattern = re.compile(r"cam\d+")

    cams = [cam for cam in os.listdir(folder) if pattern.match(cam)]

    cam_names = [cam.split('.')[0].split('_')[0] for cam in cams]

    if omni_tag != '':
        cam_names = [cam for cam in cam_names if omni_tag not in cam]
    # remove duplicates
    cam_names = list(set(cam_names))
    cam_names.sort()

    return cam_names



def get_frame_from_file(frame_path:Path) -> np.ndarray:
    """Get frame from file.
    Args:
        frame_path (str): Path to the frame.
    Returns:
        np.ndarray: Frame.
    """
    assert frame_path.is_file(), f"Tried to load {frame_path}, not a file"

    frame = cv2.imread(str(frame_path))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame  

