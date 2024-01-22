"""
Calculates the extrinsics for all cameras with environment footage

 - Loads intrinsics and sets up the opensfm directory with undistorted frames

 - Runs OpenSfM to calculate the extrinsics

 - Saves the extrinsics to a json file, one per camera
"""


from flask import Blueprint, render_template, redirect, url_for, current_app
import os
import shutil

# imports
import os
import sys
import cv2
import numpy as np
import subprocess
import pyexif
from pathlib import Path

sys.path.append('..')

from utils.log_utils import log
from utils.multiview_utils import Camera
from utils.metadata_utils import get_cam_names
from configs.arguments import get_config_dict
from utils.io_utils import write_json, load_json

opensfm_app = Blueprint('opensfm_app', __name__)

@opensfm_app.route('/opensfm/run')
def run_opensfm():
    config = current_app.config.get('CONFIG')
    force_reconstruction = config.get('force_reconstruction', False)
    num_extract = config.get('num_extract', 2)
    num_extract_omni = config.get('num_extract_omni', 30)
    skip_percent_frames = config.get('skip_percent_frames', 10) / 100
    omni_frame_ids = config.get('omni_specified_frame_ids', None)
    persp_frame_ids = config.get('persp_specified_frame_ids', None)

    try:
        config = get_config_dict()
    except:
        log.warning("No config file found. Using default values.")
        config = {}

    data_root = Path(config.get('main', {}).get('data_root', '/root/data'))
    omni_tag = config.get('calibration', {}).get('omni_tag', '360')
    force_reconstruction = config.get('calibration', 
                                    {}).get('force_reconstruction', False)
    opensfm_repo = Path(config.get('calibration', 
                                {}).get('opensfm_repo', '/OpenSfM'))
    

    # set up paths
    env_footage = data_root / 'raw_data' / 'footage'
    opensfm_data =  data_root / '0-calibration' / 'opensfm'
    opensfm_images = opensfm_data / 'images'
    extrinsics_path = data_root / '0-calibration' / 'extrinsics'

    def main():
        if ((opensfm_data / 'reconstruction.json').is_file()
            and not force_reconstruction):
            log.info("Reconstruction already exists. Skipping reconstruction.")
        
        else:
            log.info("Running SfM pipeline...")
            # Check if openSfM repository exists
            if not opensfm_repo.is_dir():
                raise ValueError(
                    "Cannot run openSFM since it has not been "
                    "cloned yet, follow instructions at:"
                    "https://opensfm.readthedocs.io/en/latest/building.html")
            
            opensfm_images.mkdir(parents=True, exist_ok=True)

            cams = get_cam_names(env_footage)
            camera_models_overrides_dict = {}
            log.info(f"Found {(cams)} in environment footage.")

            # check that selected frames file exists
            if not (extrinsics_path / 'selected_frames.json').is_file():
                log.error("No selected_frames.json found. Run camera_selection.ipynb or resort to full extrinsics calculation.")
                return 1
            else:
                selected_frames = load_json(extrinsics_path / 'selected_frames.json')
            

            for interval_name, interval_data in selected_frames.items:
                opensfm_images_interval = opensfm_images / interval_name
                opensfm_data_interval = opensfm_data / interval_name
                interval_start = interval_data['start_time']
                interval_end = interval_data['end_time']

                for camera_name in interval_data['cameras']:
                    camera = Camera(camera_name, newest=False)
                    first_frame = int(skip_percent_frames*(camera.num_frames))
                    last_frame = camera.num_frames - first_frame

                    cam_dict = {}
                    if camera.is_omni:
                        frame_ids = np.linspace(interval_start, interval_end, 
                                                num_extract_omni, dtype=int)
                        frames = camera.extract(frame_ids)
                        height, width = frames[0].shape[:-1]
                        proj_type = 'spherical'
                        cam_dict["projection_type"] = proj_type
                        for i, frame in enumerate(frames):
                            index = frame_ids[i]
                            file_path =  opensfm_images / f'{camera.name}_{index}.jpg'
                            cv2.imwrite(str(file_path), frame)
                            metadata = pyexif.ExifEditor(file_path)
                            metadata.setTag('Model', camera.name)
                            metadata.setTag('Make', camera.name)
                    else:
                        frame_ids = np.linspace(first_frame, last_frame, 
                                                num_extract, dtype=int)
                        frames = camera.extract(frame_ids)
                        K = camera.get_calib().K
                        frames = camera.undistort(frames=frames)
                        height, width = frames[0].shape[:-1]
                        focal_length = (K[0][0]+K[1][1]) / (np.max([height, width])*2)
                        proj_type ='perspective'
                        cam_dict["projection_type"] = proj_type
                        cam_dict["focal"] = focal_length

                        for i, frame in enumerate(frames):
                            index = frame_ids[i]
                            file_path =  opensfm_images_interval / f'{camera.name}_{index}.jpg'              
                            cv2.imwrite(str(file_path), frame)
                            metadata = pyexif.ExifEditor(file_path)
                            metadata.setTag('Model', camera.name)
                            metadata.setTag('Make', camera.name)

                    cam_dict["width"] = width
                    cam_dict["height"] = height
                    cam_key = f"v2 {cam}  {width} {height} perspective 0.0"
                    camera_models_overrides_dict[cam_key] = cam_dict

                # Write to file
                log.info("Adding overrides to opensfm data directory...")
                overrides_filepath = opensfm_data_interval / "camera_models_overrides.json"

                write_json(overrides_filepath, camera_models_overrides_dict)

                # Run openSfM on the data folder
                osfm_runall = opensfm_repo / 'bin' / 'opensfm_run_all'
                bash_command = f"bash {osfm_runall} {opensfm_data_interval}"
                process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

                # Wait for command to finish and get return code
                return_code = process.wait()

                log.info(f"SfM pipeline finished with return code: {return_code}")

                if return_code != 0:
                    log.error("SfM pipeline failed.")
                    return 1
    ################################ UP TO HERE ####################################
        cams = get_cam_names(env_footage, omni_tag=omni_tag)

        for cam in cams:
            camera = Camera(cam, newest=False)
            camera.calib_from_reconstruction(opensfm_data / 'reconstruction.json')
            if camera.is_calibrated():
                log.info(f"Camera {cam} calibrated.")
                camera.save_calibration()
        return 0
    
    status = main()
    return render_template('opensfm_run.html', status)

