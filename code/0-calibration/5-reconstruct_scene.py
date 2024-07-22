"""
Extract features and matches them between all the images.

Using the matches between the cubemap from the 360 frames, we reconstruct an initial sparse point cloud.

The feature and matches from the perspective frame are stored and will be used in the next step to localize the camera.

"""

"""
Output:
    - outputs: directory containing the extracted features and matches and pairs of images used for reconstruction.
    - 360-reconstruction.ply file: sparse point cloud of the initial reconstruction.
"""

# imports
import os
import sys
import cv2
import copy
import numpy as np
import pycolmap
import shutil

# Path to the code directory
BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('code')[-2]
CODEPATH = os.path.join(BASEPATH, 'code')
DATAPATH = os.path.join(BASEPATH, 'data')

sys.path.append(CODEPATH)


from matplotlib import pyplot as plt
from pathlib import Path

from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster

from configs.arguments import get_config_dict
from utils import pairs_from_sequence, pairs_from_360
from utils.multiview_utils import Camera, Calibration
from utils.metadata_utils import get_cam_names
from utils.colmap_utils import add_camera_and_image, camera_calibs_from_colmap
from utils.colmap_visualization import plot_reconstruction
from utils.log_utils import log

try:
    config = get_config_dict()
except:
    log.warning("No config file found. Using default values.")
    config = {}

data_root = Path(config.get('main', {}).get('data_root', DATAPATH))

    
RESTART = config["calibration"]["force_reconstruction"]

images = data_root / "0-calibration/images"
outputs = data_root / "0-calibration/outputs/"

sfm_pairs = outputs / "pairs-sfm.txt"
loc_pairs = outputs / "pairs-loc.txt"
sfm_dir = outputs / "sfm"
features = outputs / "features.h5"
matches = outputs / "matches.h5"
model_dir = outputs / 'full_reconstruction/'

rec_already_exists = (outputs / "full_reconstruction").exists()

if rec_already_exists and not RESTART:
    log.info("Reconstruction already exists. Skipping reconstruction.")
    sys.exit(0)

if RESTART or not rec_already_exists:
    if outputs.exists():
        # Remove all contents and the directory itself
        shutil.rmtree(outputs)
        log.info(f"Removing existing reconstruction output in: {outputs}")

outputs.mkdir(exist_ok=True)

# Feature extraction and matching configuration
feature_conf = extract_features.confs["disk"]
matcher_conf = match_features.confs["disk+lightglue"]

log.info("Starting scene reconstruction using 360 footage.")
references = [p.relative_to(images).as_posix() for p in images.rglob('*.jpg') if '360' in p.parent.name]

log.info(f"Found {len(references)} omnidirectional images.")


if RESTART or not rec_already_exists:
    log.info("Feature extraction")
    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )

    pairs_from_360.main(sfm_pairs, image_dir=images, window_size=4)

    log.info("Feature matching")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches);



    log.info("Reconstruction")
    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, features, matches, image_list=references, camera_mode=pycolmap.CameraMode.PER_FOLDER)


    model_dir.mkdir(exist_ok = True, parents = True)

    model.write(model_dir)
    model.export_PLY(outputs / "360-reconstruction.ply")
else:
    log.info(f"Loading existing 360 reconstruction from {outputs / 'full_reconstruction'}")
    model = pycolmap.Reconstruction(outputs / "full_reconstruction")

log.info(f"Initial reconstruction completed. Point cloud saved in {outputs / '360-reconstruction.ply'}")


log.info(f"Localizing static cameras")

query = [p.relative_to(images).as_posix() for p in images.rglob('*.jpg') if '360' not in p.parent.name]

if RESTART or not rec_already_exists:
    log.info("Feature extraction static camera")
    extract_features.main(
        feature_conf, images, image_list=query, feature_path=features, overwrite=True
    )
    pairs_from_exhaustive.main(loc_pairs, image_list=query, ref_list=references)

    log.info("Feature matching static camera")
    match_features.main(
        matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
    );



# camera = pycolmap.infer_camera_from_image(images / query)
ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r) is not None]
conf = {
    "estimation": {"ransac": {"max_error": 12}},
    "refinement": {"refine_focal_length": False, "refine_extra_params": False},
}

if RESTART or not rec_already_exists:
    localizer = QueryLocalizer(model, conf)
    rets = []
    logs = []
    for query_ in query:
        camera = pycolmap.infer_camera_from_image(images / query_)
        camera.params = [640, 640, 340, 0]
        ret, pose_log = pose_from_cluster(localizer, query_, camera, ref_ids, features, matches)
        ret['image'] = query_
        # add results to reconstruction
        try:
            add_camera_and_image(model, ret['camera'], ret["cam_from_world"], ret["image"])
        except ValueError as e:
            log.error(e)
    
        rets.append(ret)
        logs.append(pose_log)

        log.debug(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')

    

    log.info(f"Localizing static cameras completed.")

    log.info(f"Automatic reconstruction alignement")

    temp_model = copy.deepcopy(model)

    # Transform the model
    temp_model.transform(pycolmap.Sim3d(1, pycolmap.Rotation3d([-0.7071068, 0, 0, 0.7071068]), [0, 0, 0]))

    # Filter out outlier and use bbox to set initial z=0
    bbox = temp_model.compute_bounding_box(0.01, 0.99)
    temp_model.transform(pycolmap.Sim3d(1, pycolmap.Rotation3d([0, 0, 0, 1]), [0, 0, -temp_model.compute_bounding_box(0.01, 0.99)[0][2]]))
    temp_model.transform(pycolmap.Sim3d(100, pycolmap.Rotation3d([0, 0, 0, 1]), [0, 0, 0]))


    cameras = camera_calibs_from_colmap(images, temp_model, save=True)

    log.info("Initial extrinsic camera calibration completed.")

    # Visualize the reconstruction
    plot_reconstruction(temp_model, save_path=outputs / "360-reconstruction_w_static.html")

    log.info(f"Scene visualization completed. Results saved in {outputs / '360-reconstruction_w_static.html'}")