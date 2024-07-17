#!/usr/bin/env python3

# Source: https://github.com/mapillary/OpenSfM/pull/583/files

# Snippets to read from the colmap database taken from:
# https://github.com/colmap/colmap/blob/ad7bd93f1a27af7533121aa043a167fe1490688c /
# scripts/python/export_to_bundler.py
# scripts/python/read_write_model.py
# License is that derived from those files.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import math
import os
import sqlite3
from pathlib import Path

import networkx
import numpy as np

from opensfm import dataset
from opensfm import features
from opensfm import io
from opensfm import types

EXPORT_DIR_NAME = 'opensfm_export'


def import_cameras_images(db, data):
    cursor = db.cursor()
    cursor.execute("SELECT camera_id, model, width, height, prior_focal_length, params FROM "
                   "cameras;")
    cameras = {}
    for row in cursor:
        camera_id, camera_model_id, width, height, prior_focal, params = row
        params = np.fromstring(params, dtype=np.double)
        cam = cam_from_colmap_params(camera_model_id, width, height, params, prior_focal)
        cam.id = str(camera_id)
        cameras[camera_id] = cam

    data.save_camera_models(cameras)

    images_map = {}
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id, camera_id, filename = int(row[0]), int(row[1]), row[2]
        images_map[image_id] = (filename, camera_id)
        cam = cameras[camera_id]
        focal_ratio = cam.focal_x if isinstance(cam, types.BrownPerspectiveCamera) else cam.focal
        exif_data = {
            "make": "unknown",
            "model": "unknown",
            "width": cam.width,
            "height": cam.height,
            "projection_type": cam.projection_type,
            "focal_ratio": focal_ratio,
            "orientation": 1,
            "camera": "{}".format(camera_id),
            "skey": "TheSequence",
            "capture_time": 0.0,
            "gps": {},
        }
        data.save_exif(filename, exif_data)

    cursor.close()
    return cameras, images_map


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


def get_scale_orientation_from_affine(arr):
    # (x, y, a_11, a_12, a_21, a_22)
    a11 = arr[:, 2]
    a12 = arr[:, 3]
    a21 = arr[:, 4]
    a22 = arr[:, 5]
    scale_x = np.sqrt(a11 * a11 + a21 * a21)
    scale_y = np.sqrt(a12 * a12 + a22 * a22)
    orientation = np.arctan2(a21, a11)
    # shear = np.arctan2(-a12, a22) - orientation
    scale = (scale_x + scale_y) / 2
    return scale, orientation


def import_features(db, data, image_map, camera_map):
    cursor = db.cursor()
    cursor.execute("SELECT image_id, rows, cols, data FROM keypoints;")
    keypoints = {}
    colors = {}
    for row in cursor:
        image_id, n_rows, n_cols, arr = row
        filename, camera_id = image_map[image_id]
        cam = camera_map[camera_id]

        arr = np.fromstring(arr, dtype=np.float32).reshape((n_rows, n_cols))

        rgb = data.load_image(filename).astype(np.float32)
        xc = np.clip(arr[:, 1].astype(int), 0, rgb.shape[0] - 1)
        yc = np.clip(arr[:, 0].astype(int), 0, rgb.shape[1] - 1)
        colors[image_id] = rgb[xc, yc, :]

        arr[:, :2] = features.normalized_image_coordinates(arr[:, :2], cam.width, cam.height)
        if n_cols == 4:
            x, y, s, o = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        elif n_cols == 6:
            x, y = arr[:, 0], arr[:, 1]
            s, o = get_scale_orientation_from_affine(arr)
        elif n_cols == 2:
            x, y = arr[:, 0], arr[:, 1]
            s = np.zeros_like(x)
            o = np.zeros_like(x)
        else:
            raise ValueError
        s = s / max(cam.width, cam.height)
        keypoints[image_id] = np.vstack((x, y, s, o)).T

    cursor.execute("SELECT image_id, rows, cols, data FROM descriptors;")
    for row in cursor:
        image_id, n_rows, n_cols, arr = row
        filename, _ = image_map[image_id]
        descriptors = np.fromstring(arr, dtype=np.uint8).reshape((n_rows, n_cols))
        kp = keypoints[image_id]
        data.save_features(filename, kp, descriptors, colors[image_id])

    cursor.close()
    return keypoints


def import_matches(db, data, image_map):
    cursor = db.cursor()
    min_matches = 1
    cursor.execute("SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;", (min_matches,))

    matches_per_im1 = {m[0]: {} for m in image_map.values()}

    for row in cursor:
        pair_id = row[0]
        inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2)
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        image_name1 = image_map[image_id1][0]
        image_name2 = image_map[image_id2][0]
        matches_per_im1[image_name1][image_name2] = inlier_matches

    for image_name1, matches in matches_per_im1.items():
        data.save_matches(image_name1, matches)

    cursor.close()


def import_cameras_reconstruction(path_cameras):
    """
    Imports cameras from a COLMAP reconstruction text file
    """
    r_cams = {}
    mapping = {'FULL_OPENCV': 6, 'RADIAL': 3, 'RADIAL_FISHEYE': 9}
    with io.open_rt(path_cameras) as fin:
        for row in fin:
            if row[0] == '#':
                continue
            row = row[:-1].split(' ')
            camera_id = row[0]
            camera_model = row[1]
            width = int(row[2])
            height = int(row[3])
            params = [float(p) for p in row[4:]]
            camera_model_id = mapping[camera_model]
            cam = cam_from_colmap_params(camera_model_id, width, height, params)
            cam.id = camera_id
            r_cams[camera_id] = cam
    return r_cams


def cam_from_colmap_params(camera_model_id, width, height, params, prior_focal=1):
    """
    Helper function to map from colmap parameters to an OpenSfM camera
    """
    mapping = {3: 'perspective', 6: 'brown', 9: 'fisheye'}
    projection_type = mapping[camera_model_id]
    normalizer = max(width, height)
    if projection_type == 'perspective':
        cam = types.PerspectiveCamera()
        cam.focal = params[0] / normalizer if prior_focal else 0.85
        cam.k1 = params[3]
        cam.k2 = params[4]
    elif projection_type == 'brown':
        cam = types.BrownPerspectiveCamera()
        cam.focal_x = params[0] / normalizer if prior_focal else 0.85
        cam.focal_y = params[1] / normalizer if prior_focal else 0.85
        cam.c_x = (params[2] - (width - 1) * 0.5) / normalizer
        cam.c_y = (params[3] - (height - 1) * 0.5) / normalizer
        cam.k1, cam.k2, cam.p1, cam.p2, cam.k3 = params[4:9]
    else:  # projection_type == 'fisheye'
        cam = types.FisheyeCamera()
        cam.focal = params[0] / normalizer if prior_focal else 0.85
        cam.k1 = params[3]
    cam.width = width
    cam.height = height
    return cam


def import_shots_reconstruction(path_shots, camera_map, keypoints, points3D):
    """
    Reads the points.txt from colmap, which contains two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    With this, we build the OpenSfM shots and a partial tracks graph.
    """
    tracks_graph = networkx.Graph()
    shots = {}
    with io.open_rt(path_shots) as fin:
        while True:
            row = fin.readline().strip()
            if not row:  # None or len==0
                break
            if row[0] == '#':
                continue

            row = row.split(' ')
            colmap_shot_id = int(row[0])
            q = np.array(tuple(map(float, row[1:5])))
            t = np.array(tuple(map(float, row[5:8])))
            colmap_camera_id = row[8]
            image = row[9]  # filename / key

            shot = types.Shot()
            shot.pose = types.Pose(rotation=quaternion_to_angle_axis(q), translation=t)
            shot.camera = camera_map[colmap_camera_id]
            shot.id = image
            shots[shot.id] = shot

            row = fin.readline().strip().split(' ')
            xys = np.column_stack([tuple(map(float, row[0::3])), tuple(map(float, row[1::3]))])
            xys = features.normalized_image_coordinates(xys,
                                                        shot.camera.width,
                                                        shot.camera.height)
            point3D_ids = np.array(tuple(map(int, row[2::3])))

            point2d_idx = 0
            for point3D_id, xy in zip(point3D_ids, xys):
                if point3D_id == -1:
                    continue
                kp = keypoints[colmap_shot_id][point2d_idx]
                s = kp[2]
                tracks_graph.add_node(str(image), bipartite=0)
                tracks_graph.add_node(str(point3D_id), bipartite=1)
                tracks_graph.add_edge(str(image),
                                      str(point3D_id),
                                      feature=(float(xy[0]), float(xy[1])),
                                      feature_id=point2d_idx,
                                      feature_scale=float(s),
                                      feature_color=points3D[point3D_id].color)
                point2d_idx += 1

    return shots, tracks_graph


def import_points_reconstruction(path_points):
    points3d = {}

    with io.open_rt(path_points) as fin:
        for row in fin:
            if row[0] == '#':
                continue
            row = row[:-1].split(' ')
            p = types.Point()
            p.id = int(row[0])
            p.coordinates = tuple(map(float, row[1:4]))
            p.color = tuple(map(int, row[4:7]))
            points3d[p.id] = p
    return points3d


def quaternion_to_angle_axis(quaternion):
    if quaternion[0] > 1:
        quaternion = quaternion / np.linalg.norm(quaternion)
    qw, qx, qy, qz = quaternion
    s = max(0.001, math.sqrt(1 - qw * qw))
    x = qx / s
    y = qy / s
    z = qz / s
    angle = 2 * math.acos(qw)
    return [angle * x, angle * y, angle * z]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert COLMAP database to OpenSfM dataset')
    parser.add_argument('database', help='path to the database to be processed')
    parser.add_argument('images', help='path to the images')
    args = parser.parse_args()

    p_db = Path(args.database)
    export_folder = p_db.parent / EXPORT_DIR_NAME
    export_folder.mkdir(exist_ok=True)
    images_path = export_folder / 'images'
    if not images_path.exists():
        os.symlink(args.images, images_path, target_is_directory=True)

    data = dataset.DataSet(export_folder)
    db = sqlite3.connect(p_db.as_posix())
    camera_map, image_map = import_cameras_images(db, data)
    keypoints = import_features(db, data, image_map, camera_map)
    import_matches(db, data, image_map)

    rec_cameras = p_db.parent / 'cameras.txt'
    rec_points = p_db.parent / 'points3D.txt'
    rec_images = p_db.parent / 'images.txt'
    if rec_cameras.exists() and rec_images.exists() and rec_points.exists():
        cameras = import_cameras_reconstruction(rec_cameras)
        points3D = import_points_reconstruction(rec_points)
        shots, tracks_graph = import_shots_reconstruction(rec_images, cameras, keypoints, points3D)
        data.save_tracks_graph(tracks_graph)

        reconstruction = types.Reconstruction()
        reconstruction.cameras = cameras
        reconstruction.shots = shots
        reconstruction.points = points3D
        data.save_reconstruction([reconstruction])
    else:
        print("Didn't find reconstruction files in text format")

    db.close()