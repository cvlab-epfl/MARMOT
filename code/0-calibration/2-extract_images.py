"""
Extracts images from all video files and places them in the images file.

For 360 footage it axtracts images at regular intervals. These images are then
split into perspective images.

"""

"""
Calculates the extrinsics for all cameras with environment footage

 - Loads intrinsics and sets up the opensfm directory with undistorted frames

 - Runs OpenSfM to calculate the extrinsics

 - Saves the extrinsics to a json file, one per camera
"""

# extrinsics calculation parameters

force_reconstruction = False
num_extract = 2
num_extract_omni = 20
interval_omni = 60
skip_percent_frames = 0.10

# imports
import os
import sys
import cv2
import numpy as np
import subprocess
import pyexif
from pathlib import Path
import py360convert
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Path to the code directory
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('code')[-2]
CODEPATH = os.path.join(BASEPATH, 'code')
DATAPATH = os.path.join(BASEPATH, 'data')
sys.path.append(CODEPATH)

from utils.log_utils import log
from utils.multiview_utils import Camera
from utils.metadata_utils import get_cam_names
from configs.arguments import get_config_dict
from utils.io_utils import write_json


omni_frame_ids = np.linspace(1200, 2700, 1500 // 500, dtype=int)
persp_frame_ids = [100]
omni_frame_min = 1200
omni_frame_max = 2700


try:
    config = get_config_dict()
except:
    log.warning("No config file found. Using default values.")
    config = {}

data_root = Path(config.get('main', {}).get('data_root', DATAPATH))
omni_tag = config.get('calibration', {}).get('omni_tag', '360')
# set up paths
env_footage = data_root / 'raw_data' / 'footage'
# opensfm_data =  data_root / '0-calibration' / 'opensfm'
images_dir = data_root / '0-calibration' / 'images'
output_dir = data_root / '0-calibration' / 'outputs'

output_dir.mkdir(parents=True, exist_ok=True)

REEXTRACT = True
views_list = ['forward', 'right', 'back', 'left', 'up', 'down']
const = 1 / np.sqrt(2)
quat_list = [
    [1, 0, 0, 0],
    [-const, 0, const, 0],
    [0, 0, 1, 0],
    [const, 0, const, 0],
    [const, 0, 0, const],
    [-const, 0, 0, const]
]
rig_adjuster_config = [
        {
            "ref_camera_id":1,
            "cameras":
            []
        }
    ]

def process_omni_frame(frame, camera:Camera, index, face_w, rig_adjuster_config):
    perspectives = py360convert.e2c(frame, face_w=face_w, cube_format='list')
    
    
    for j, image in enumerate(perspectives):
        # if views_list[j] in ['back', 'up', 'down']:
        #     continue
        cam_id = j + 1
        views_folder = images_dir / f'{camera.name}_view_{views_list[j]}'
        views_folder.mkdir(parents=True, exist_ok=True)
        file_path =  views_folder / f'{index}.jpg' 
        cv2.imwrite(str(file_path), image)
        # metadata = pyexif.ExifEditor(file_path)
        # metadata.setTag('Model', camera.name)
        # metadata.setTag('Make', camera.name)

        config_entry = {
            "camera_id": cam_id,
            "image_prefix": f"{camera.name}_view_{views_list[j]}",
            "cam_from_rig_rotation": quat_list[j],
            "cam_from_rig_translation": [0, 0, 0]
        }
        if config_entry not in rig_adjuster_config[0]["cameras"]:
            rig_adjuster_config[0]["cameras"].append(config_entry)
    return rig_adjuster_config
    

def process_perspective_frame(frame, camera:Camera, index):
    views_folder = images_dir / f'{camera.name}_view_{views_list[0]}'
    views_folder.mkdir(parents=True, exist_ok=True)
    file_path =  views_folder / f'{index}.jpg' 
    cv2.imwrite(str(file_path), frame)
    # metadata = pyexif.ExifEditor(file_path)
    # metadata.setTag('Model', camera.name)
    # metadata.setTag('Make', camera.name)

def main():
    log.info("Extracting Images...")
    
    if REEXTRACT:
        if images_dir.is_dir():
            log.info("Removing existing images directory.")
            subprocess.run(f"rm -r {images_dir}", shell=True)

    images_dir.mkdir(parents=True, exist_ok=True)

    cams = get_cam_names(env_footage)
    camera_models_overrides_dict = {}
    log.info(f"Found {(cams)} in environment footage.")

    for cam in cams:
        camera = Camera(cam, newest=False)
        
        first_frame = int(skip_percent_frames*(camera.num_frames))
        last_frame = camera.num_frames - first_frame

        cam_dict = {}
        if camera.is_omni:
            num_extract_omni = (last_frame - first_frame) // 15
            frame_ids = np.linspace(first_frame, last_frame, 
                                    num_extract_omni, dtype=int)
            if omni_frame_ids is not None:
                frame_ids = omni_frame_ids
            # elif omni_frame_min is not None and omni_frame_max is not None:
            #     frame_ids = np.linspace(omni_frame_min, omni_frame_max, 
            #                             num_extract_omni, dtype=int)
            all_frame_ids = frame_ids
            batch_size = 10
            with ThreadPoolExecutor() as executor:
                futures = []
                for batch_start in tqdm(range(0, len(all_frame_ids), batch_size), desc=f"Extracting {cam} frames"):
                    batch_frame_ids = all_frame_ids[batch_start:batch_start+batch_size]
                    frames = camera.extract(batch_frame_ids)
                    height, width = frames[0].shape[:-1]
                    proj_type = 'perspective'
                    cam_dict["projection_type"] = proj_type
                    face_w = width // 4
                    cam_dict["focal"] = face_w / 2
                    cam_dict["width"] = face_w
                    cam_dict["height"] = face_w
                    for i, frame in enumerate(frames):
                        index = batch_frame_ids[i]
                        futures.append(executor.submit(process_omni_frame, frame, camera, index, face_w, rig_adjuster_config))

                for future in as_completed(futures):
                    pass

        else:
            frame_ids = np.linspace(first_frame, last_frame, 
                                    num_extract, dtype=int)
            if persp_frame_ids is not None:
                frame_ids = persp_frame_ids
            frames = camera.extract(frame_ids)
            # K = camera.get_calib().K
            # frames = camera.undistort(frames=frames)
            height, width = frames[0].shape[:-1]
            # focal_length = (K[0][0]+K[1][1]) / (np.max([height, width])*2)
            proj_type ='perspective'
            # cam_dict["projection_type"] = proj_type
            # cam_dict["focal"] = focal_length

            for i, frame in tqdm(enumerate(frames), desc=f"Extracting {cam} frames"):
                index = frame_ids[i]
                process_perspective_frame(frame, camera, index)

            cam_dict["width"] = width
            cam_dict["height"] = height
            cam_key = f"v2 {cam}  {width} {height} perspective 0.0"
            camera_models_overrides_dict[cam_key] = cam_dict

    # save rig_adjuster_config
    write_json(CODEPATH / "configs" / "rig_config.json", rig_adjuster_config)

if __name__ == "__main__":
    main()