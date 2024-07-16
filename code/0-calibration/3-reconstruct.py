import tqdm, tqdm.notebook

tqdm.tqdm = tqdm.notebook.tqdm  # notebook-friendly progress bars
from pathlib import Path
import numpy as np
import os
import sys
import pycolmap
# Path to the code directory
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('code')[-2]
CODEPATH = os.path.join(BASEPATH, 'code')
DATAPATH = os.path.join(BASEPATH, 'data')
sys.path.append(CODEPATH)


from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from utils.log_utils import log
from utils.multiview_utils import Camera, Calibration
from utils.metadata_utils import get_cam_names
from configs.arguments import get_config_dict
from utils.io_utils import write_json

images = Path(DATAPATH) / "0-calibration" / "images"
outputs = Path(DATAPATH) / "0-calibration" / "output"
omni_tag = "360"

import yaml
import subprocess

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_colmap_command(command, **kwargs):
    cmd = ['colmap', command]
    for key, value in kwargs.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                cmd.append(f'--{sub_key}={sub_value}')
        else:
            cmd.append(f'--{key}={value}')
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def localise_camera(camera:Camera, config_path:Path):
    config = load_config(config_path)
    query = [img_name for img_name in images if camera.name in img_name and not '360' in img_name]

    extract_features.main(
        feature_conf, images, image_list=[query], feature_path=features, overwrite=True
    )
    pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
    match_features.main(
        matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
    );

def main(config_path= Path(CODEPATH) / "configs" / 'colmap_config.yaml'):
    config = load_config(config_path)

    outputs.mkdir(exist_ok=True, parents=True)
    sfm_dir = outputs / "sfm"
    sfm_dir.mkdir(exist_ok=True, parents=True)

    # Feature Extraction
    run_colmap_command('feature_extractor', **config['feature_extractor'])

    # Matching
    run_colmap_command('sequential_matcher', **config['matcher'])

    # Reconstruction
    run_colmap_command('hierarchical_mapper', **config['mapper'])

    # Retriangulate
    run_colmap_command('point_triangulator', **config['mapper'])

    # Bundle Adjustment
    run_colmap_command('bundle_adjuster', **config['bundle_adjustment'])

    # Rig Bundle Adjustment
    run_colmap_command('rig_bundle_adjuster', **config['rig_bundle_adjustment'])

    # Model Orientation Aligner
    run_colmap_command('model_orientation_aligner', **config['model_orientation_aligner'])

    return 1
    cams = get_cam_names(DATAPATH / "raw_data" / "footage")
    log.info(f"Found {(cams)} in environment footage.")

    for cam in cams:
        camera = Camera(cam, newest=False)
        extrinsics = localise_camera()
        calib = camera.get_calib()
        camera.set_calib(Calibration())

    

if __name__ == '__main__':
    main()