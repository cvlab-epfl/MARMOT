import numpy as np
import cv2
import os
import pickle
import json
import gzip

__all__ = [
    'load_features_and_matches',
    'return_unposed',
    'denormalized_image_coordinates',
    'draw_features',
    'global_registration'
]

def return_unposed(reconstruction_data):
    if reconstruction_data is None:
        print("Reconstruction not found. Please run 0-calibration/2-extrinsics.py first.")
        return None
    
    list_of_cams = set([key for key in reconstruction_data['cameras'].keys()])
    print(list_of_cams)
    for shot in reconstruction_data['shots'].values():
        print(shot)
        if shot['camera'] in list_of_cams:
            list_of_cams.remove(shot['camera'])
    
    return list(list_of_cams)

def denormalized_image_coordinates(
        norm_coords: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        size = max(width, height)
        p = np.empty((len(norm_coords), 2))
        # handle the case where only a single point is passed
        if len(norm_coords.shape) == 1:
            norm_coords = norm_coords.reshape(1, -1)

        p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
        p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
        return p


def draw_features(img, points, colour = (0, 0, 255)):
    if colour != (0, 0, 255):
        point_size = 5
    else:
        point_size = 2
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), point_size, colour, -1)
    return img



# multiview_calib implementation

def global_registration(upload_folder):
    """
    Perform global registration to get global poses
    """
    setup_path = os.path.join(upload_folder, 'setup.json')
    ba_poses_path = os.path.join(upload_folder, 'ba_poses.json')
    ba_points_path = os.path.join(upload_folder, 'ba_points.json')
    landmarks_path = os.path.join(upload_folder, 'landmarks.json')
    landmarks_global_path = os.path.join(upload_folder, 'landmarks_global.json')
    filenames_path = os.listdir(os.path.join(upload_folder, 'filenames.json'))

    os.system(f'python triangulate_image_points.py -p {ba_poses_path} -l {landmarks_path} --dump_images')


    os.system(f'python global_registration.py -s {setup_path} -ps {ba_poses_path} -po {ba_points_path} -l {landmarks_path} -lg {landmarks_global_path} -f {filenames_path} --dump_images')

    return '', 204



def load_features_and_matches(img_path, upload_folder):
    features_path = os.path.join(upload_folder, 'features')
    matches_path = os.path.join(upload_folder, 'matches')
    img_name = os.path.basename(img_path)
    features = np.load(os.path.join(features_path, img_name + '.features.npz'))

    with gzip.open(os.path.join(matches_path, img_name + '_matches.pkl.gz'), 'rb') as f:
        matches = pickle.load(f)

    return features, matches


def compute_relative_poses(upload_folder):
    """
    Compute relative poses between cameras using multiview_calib
    """
    setup_path = os.path.join(upload_folder, 'setup.json')
    intrinsics_path = os.path.join(upload_folder, 'intrinsics.json')
    landmarks_path = os.path.join(upload_folder, 'landmarks.json')
    filenames_path = os.listdir(os.path.join(upload_folder, 'filenames.json'))

    os.system(f"python compute_relative_poses.py -s {setup_path} -i {intrinsics_path} -l {landmarks_path} -f {filenames_path} --dump_images")

    # add visualisation to the rendering
    pass

def compute_relative_poses_robust(upload_folder):
    """
    Compute relative poses between cameras using multiview_calib
    """
    setup_path = os.path.join(upload_folder, 'setup.json')
    intrinsics_path = os.path.join(upload_folder, 'intrinsics.json')
    landmarks_path = os.path.join(upload_folder, 'landmarks.json')
    filenames_path = os.listdir(os.path.join(upload_folder, 'filenames.json'))
    

    os.system(f'python compute_relative_poses_robust.py -s {setup_path} -i {intrinsics_path} -l {landmarks_path} -f {filenames_path} --dump_images')

    pass

def concatenate_relative_poses(upload_folder):
    """
    Concatenate relative poses to get global poses
    """
    setup_path = os.path.join(upload_folder, 'setup.json')
    relative_poses_path = os.path.join(upload_folder, 'relative_poses.json')

    os.system(f'python concatenate_relative_poses.py -s {setup_path} -r {relative_poses_path} --dump_images')

    pass

def bundle_adjustment(upload_folder):
    """
    Perform bundle adjustment to refine the global poses
    """
    setup_path = os.path.join(upload_folder, 'setup.json')
    intrinsics_path = os.path.join(upload_folder, 'intrinsics.json')
    poses_path = os.path.join(upload_folder, 'poses.json')
    landmarks_path = os.path.join(upload_folder, 'landmarks.json')
    filenames_path = os.listdir(os.path.join(upload_folder, 'filenames.json'))
    ba_config_path = os.path.join(upload_folder, 'ba_config.json')

    os.system(f'python bundle_adjustment.py -s {setup_path} -i {intrinsics_path} -e {poses_path} -l {landmarks_path} -f {filenames_path} --dump_images -c {ba_config_path}')

    pass

