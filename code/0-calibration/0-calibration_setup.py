"""
Sets up the environment to run the calibration pipeline.
- Verifies that opensfm is installed
- Makes sure that the config file is valid
- Confirms found calibration footage
- Confirms found environment footage
"""



import sys
import os
import cv2

sys.path.append('..')

from pathlib import Path

from utils.log_utils import log
from utils.metadata_utils import get_cam_names
from configs.arguments import get_config_dict

def main():
    try:
        config = get_config_dict()
        data_root = Path(config.get('main', {}).get('data_root', '/root/data'))
        root_code = Path(config.get('main', {}).get('code_root', '/root/code')) 
        data_source = config.get('main', {}).get('data', None)
        if data_source is not None:
            data_source = Path(data_source) / 'raw_data'
    except FileNotFoundError:
        log.warning("No config file found.")
        data_root = '/root/data'
        data_source = None

    calibration_dir = data_root /'0-calibration'
    raw_dir = data_root / 'raw_data'


    # make raw data directory
    if not raw_dir.is_dir():
        log.info("Creating raw data directory.")
        raw_dir.mkdir(parents=True, exist_ok=True)


    # ensure that calibration footage is valid
    checkerboard_path = raw_dir / 'calibration'

    log.debug("Data Source: {}".format(data_source))


    if data_source is not None:
        # unlink if symlink already exists
        if checkerboard_path.is_symlink():
            checkerboard_path.unlink()
        # check target exists
        if not (data_source / 'calibration').is_dir():
            log.error("Checkerboard footage not found. "
                      "Please check your config file.")
            sys.exit()
        checkerboard_path.symlink_to(data_source / 'calibration')

    elif not checkerboard_path.is_dir():
        log.error("Checkerboard footage not found. "
                  "Please check your config file.")
        sys.exit()

    cam_names = get_cam_names(checkerboard_path)
    log.info("Found footage for cameras: {}".format(cam_names))
    for vid in sorted(checkerboard_path.iterdir()):
        for cam in cam_names:
            if cam in vid.name:
                cv2.VideoCapture(str(checkerboard_path / vid))
                log.info("Found footage for camera: {}".format(cam))
                break
    
    # ensure that environment footage is valid
    footage_path = raw_dir / 'footage'

    if data_source is not None:
        # unlink if symlink already exists
        if footage_path.is_symlink():
            footage_path.unlink()
        log.debug("Footage path not found. Symlinking to data source.")
        footage_path.symlink_to(data_source / 'footage')

    elif not footage_path.is_dir():
        log.error("Footage path not found. Please check your config file.")
        sys.exit()
        
    cam_names = get_cam_names(footage_path)
    log.info("Found footage for cameras: {}".format(cam_names))
    for vid in sorted(footage_path.iterdir()):
        for cam in cam_names:
            if cam in vid.name:
                cv2.VideoCapture(str(footage_path / vid))
                log.info("Found footage for camera: {}".format(cam))
                break

    # create opensfm config file
    osfm_dir = calibration_dir / 'opensfm' 
    osfm_config = osfm_dir / 'config.yaml'

    osfm_dir.mkdir(parents=True, exist_ok=True)

    if osfm_config.is_file():
        log.info("Found config file.")
    else:
        log.warning("No config file found. Creating default config file.")
        os.system(f"cp /root/opensfm_config.yaml {osfm_config}")

    # make calibration directory
    calibs_dir = calibration_dir / 'calibs'
    calibs_dir.mkdir(parents=True, exist_ok=True)

    visu_dir = calibration_dir / 'visualisation'
    visu_dir.mkdir(parents=True, exist_ok=True)

    pass


if __name__=="__main__":
    main()
