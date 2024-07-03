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
outputs = Path(DATAPATH) / "outputs"
omni_tag = "360"

def main():
    # define paths to output files
    sfm_pairs = outputs / "pairs-sfm.txt"
    loc_pairs = outputs / "pairs-loc.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    # clear output directory
    if not outputs.exists():
        outputs.mkdir()
    else:
        for p in outputs.iterdir():
            if p.is_file():
                p.unlink()

    # configurations
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superpoint+lightglue"]

    # reference images (360)
    references = [p.relative_to(images).as_posix() for p in (images).iterdir() if omni_tag in p.name]

    # extract features
    extract_features.main(
        feature_conf, images, image_list=references, feature_path=features
    )

    # generate pairs
    pairs_from_exhaustive.main(sfm_pairs, image_list=references)

    # match features
    match_features.main(
        matcher_conf, sfm_pairs, features=features, matches=matches
    )
    opts = dict(camera_model = "SIMPLE_RADIAL")#, camera_params =','.join(map(str, (256, 256, 256, 0))))

    model = reconstruction.main(
        sfm_dir, images, sfm_pairs, features, matches, image_list=references, image_options = opts
    )
    # pose cameras
    cams = get_cam_names(images, omni_tag=omni_tag)

    for cam in cams:
        print(f"Processing camera {cam}")
        camera = Camera(cam, newest=False)      
        query = [p.relative_to(images).as_posix() for p in (images).iterdir() if omni_tag not in p.name and cam in p.name][0]
        print(query)
        camera_colmap = pycolmap.infer_camera_from_image(images / query)

        extract_features.main(
            feature_conf, images, image_list=[query], feature_path=features, overwrite=True
        )
        pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references)
        match_features.main(
            matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
        )
        ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r) is not None]
        conf = {
            "estimation": {"ransac": {"max_error": 12}},
            "refinement": {"refine_focal_length": True, "refine_extra_params": True},
        }
        localizer = QueryLocalizer(model, conf)
        ret, _ = pose_from_cluster(localizer, query, camera_colmap, ref_ids, features, matches)
        pose = ret['cam_from_world']

        print(f'found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')

        camera.set_calib(Calibration(
                R=pose.rotation.matrix(), 
                T=pose.translation, view_id=camera.calibration.view_id, 
                K=camera.calibration.K, 
                K_new=camera.calibration.K_new, 
                dist=camera.calibration.dist,
                size = camera.calibration.size))
        if camera.is_calibrated():
            log.info(f"Camera {cam} calibrated.")
            camera.save_calibration()

if __name__ == '__main__':
    main()