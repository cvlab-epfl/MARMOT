"""Generates the intrinsics for the cameras located in the calibration footage folder.
The footage should be of a checkerboard pattern. The checkerboard pattern should be
placed in the center of the frame and should be visible in the entire frame. The
checkerboard pattern should be at least 7x6 squares. 
"""

import os
import sys
import cv2
import argparse
import numpy as np

sys.path.append('..')

from pathlib import Path

from utils.arguments import Arguments
from configs.arguments import get_config_dict
from utils.log_utils import log
from utils.multiview_utils import Camera, Calibration
from utils.metadata_utils import get_cam_names
from utils.intrinsics_utils import compute_intrinsics


# load arguments for calibration
args = Arguments().parse()

try:
    config = get_config_dict()
except:
    log.warning("No config file found. Using default values.")
    config = {}

data_root = Path(config.get('main', {}).get('data_root', '/root/data'))
omni_tag = config.get('calibration', {}).get('omni_tag', '360')
force_reextract = config.get('calibration', {}).get('force_intrinsics', False)

frame_step = 2


def main():
    log.info(f"Arguments:")
    log.info("--------------------")
    log.info(f"Data root: {data_root}")
    log.info(f"Omni tag: {omni_tag}")
    log.info(f"Force reextract: {force_reextract}")
    log.info(f"Frame step: {frame_step}")

    # set paths to folders
    calib_footage = data_root / 'raw_data' / 'calibration'
    assert calib_footage.is_dir(), ("Checkerboard footage not found. "
                                    "Please place checkerboard footage in the "
                                    f"folder: {calib_footage}")

    frames_dir = (data_root / '0-calibration' / 'frames')
    frames_dir.mkdir(parents=True, exist_ok=True)

    intrinsics_visu = (data_root / '0-calibration' 
                       / 'visualisation' / 'intrinsics')
    intrinsics_visu.mkdir(parents=True, exist_ok=True)

    calib_cams = get_cam_names(calib_footage, omni_tag=omni_tag)
    log.info(f"Found calibration footage for cameras: {calib_cams}")

    footage = data_root / 'raw_data' / 'footage'
    footage_cams = get_cam_names(footage, omni_tag=omni_tag)
    log.info(f"Found environment footage for cameras: {footage_cams}")

    temp_intrinsics = None

    for cam in calib_cams:
        log.info(f"Extracting frames from {cam}")
        camera = Camera(cam, newest=False)
        cam_dir = frames_dir / camera.name
        if not cam_dir.is_dir():
            log.info(f"Creating frames folder for {cam}")
            cam_dir.mkdir(parents=True, exist_ok=True)
            
        if len(os.listdir(cam_dir)) == 0:
            vid_dict = camera.index_videos()

            frame_ids = [i for i in range(
                list(vid_dict['calibration'].values())[0]['start'], 
                list(vid_dict['calibration'].values())[0]['end'] - 1, 
                frame_step)]

            frames = camera.extract(frame_ids, mode = 'calibration')
            camera.save(cam_dir, frames = frames)

        log.info(f"Computing intrinsics for camera {cam}")
        cam_visu_dir = intrinsics_visu / camera.name
        intrinsics = compute_intrinsics(cam_dir, cam_visu_dir, args)

        temp_intrinsics = intrinsics
        calibration = Calibration(  
            K = np.array(intrinsics['K']), 
            K_new = np.array(intrinsics['K_new']),
            dist = np.array(intrinsics['dist']), 
            view_id = camera.name.split('m')[1],
            size = camera.get_frame_size()
            )

        camera.save_calibration(calibration)

        if cam in footage_cams:
            footage_cams.pop(footage_cams.index(cam))
            log.info(f"Intrinsics for {camera.name} calculated.")
            
    for cam in footage_cams:
        log.warning(f"""WARNING: {cam} is not in the calibration footage.
                Please add it to the calibration footage folder.
                For now we will revert to intrinsics from the last camera.""")
        
        camera = Camera(cam, newest=False)

        if temp_intrinsics is None:
            log.warning("""No intrinsics found for any camera.
                        Please add checkerboard footage to the 
                        footage/calibration folder.""")
            return 1
        calibration = Calibration(
            K = np.array(temp_intrinsics['K']), 
            K_new = np.array(temp_intrinsics['K_new']),
            dist = np.array(temp_intrinsics['dist']), 
            view_id = camera.name.split('m')[1],
            size = camera.get_frame_size()
            )

        camera.save_calibration(calibration)

    return 0

if __name__ == '__main__':
    main()
        

