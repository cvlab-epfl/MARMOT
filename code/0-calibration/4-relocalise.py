"""
Script to relocalise the stationary cameras (not omnidirectional) found in the
footage folder.


Loads the reconstruction and performs a feature extraction and matching step on
the new images.

TODO
"""


model_dir = outputs / 'full_reconstruction/'

model_dir.mkdir(exist_ok = True, parents = True)
if RESTART:
    model.write(model_dir)

# load reconstruction

model = pycolmap.Reconstruction(model_dir)

query = [p.relative_to(images).as_posix() for p in images.rglob('*.jpg') if '360' not in p.parent.name]

extract_features.main(
    feature_conf, images, image_list=query, feature_path=features, overwrite=True
)
pairs_from_exhaustive.main(loc_pairs, image_list=query, ref_list=references)


match_features.main(
    matcher_conf, loc_pairs, features=features, matches=matches, overwrite=True
);


import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from utils.colmap_utils import add_camera_and_image, camera_calibs_from_colmap


# camera = pycolmap.infer_camera_from_image(images / query)
ref_ids = [model.find_image_with_name(r).image_id for r in references if model.find_image_with_name(r) is not None]
conf = {
    "estimation": {"ransac": {"max_error": 12}},
    "refinement": {"refine_focal_length": False, "refine_extra_params": False},
}
localizer = QueryLocalizer(model, conf)
rets = []
logs = []

for query_ in query:
    camera = pycolmap.infer_camera_from_image(images / query_)
    camera.params = [640, 640, 340, 0]
    ret, log = pose_from_cluster(localizer, query_, camera, ref_ids, features, matches)
    ret['image'] = query_
    # add results to reconstruction
    try:
        add_camera_and_image(model, ret['camera'], ret["cam_from_world"], ret["image"])
    except ValueError as e:
        print(e)

    rets.append(ret)
    logs.append(log)


camera_calibs_from_colmap(images, model, save=False)