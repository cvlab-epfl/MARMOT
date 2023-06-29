import numpy as np
import cv2
import os
import copy
from scipy.spatial.transform import Rotation as R
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from collections import namedtuple
from utils.log_utils import log
from typing import List, Tuple, Union, Optional, Dict


def project_to_ground_plane_cv2(img: np.ndarray, H: np.ndarray, 
                                homography_output_size: tuple) -> np.ndarray:
    """Project image to ground plane using OpenCV.

    Args:
        img (np.ndarray): Image to be projected.
        H (np.ndarray): Homography matrix.
        homography_output_size (tuple): Size of the output image.

    Returns:
        np.ndarray: Projected image.
    """

    # Check input arguments
    if (not isinstance(img, np.ndarray) or 
            not isinstance(H, np.ndarray) or 
            not isinstance(homography_output_size, tuple)):
        raise TypeError("img, H, and homography_output_size must be of type "
                        "np.ndarray, np.ndarray, and tuple, respectively.")

    if (img.ndim != 3 or 
            H.ndim != 2 or 
            H.shape != (3, 3) or 
            len(homography_output_size) != 2):
        raise ValueError("Invalid shapes for input arrays or tuple elements.")

    # Convert the output size tuple to integers
    h_grid, w_grid = map(int, homography_output_size)

    try:
        # Generate the meshgrid
        xp_dist, yp_dist = np.meshgrid(np.arange(w_grid), np.arange(h_grid))
        homogenous = np.stack([xp_dist, yp_dist, 
                               np.ones((h_grid, w_grid))]).reshape(3, -1)

        # Compute the homography
        map_ind = H.dot(homogenous)
        map_ind = map_ind.astype(np.float32)

        # Compute the remap arrays
        map_x, map_y = map_ind[:-1]/map_ind[-1]
        map_x = map_x.reshape(h_grid, w_grid).astype(np.float32)
        map_y = map_y.reshape(h_grid, w_grid).astype(np.float32)

        # Remap the image
        ground_image = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

        # Make sure the output has the same data type as the input
        if img.dtype != ground_image.dtype:
            ground_image = ground_image.astype(img.dtype)

        return ground_image

    except Exception as e:
        raise RuntimeError(f"Error occurred during processing: {e}")



def point_in_polygon(x:np.ndarray, corners:list) -> bool:
    """
    Determines whether a point lies within a cone spanned by a set of 
    corners and their corresponding rays to the camera.

    Args:
        x (np.ndarray): Point to check.
        corners (list): List of corners of the polygon.

    Returns:
        bool: Whether the point lies within the polygon.
    """
    x_cam = Point(x)

    # create a polygon from the corners
    poly = Polygon(corners)

    # check whether the point lies within the polygon
    return poly.contains(Point(x_cam))


def update_reconstruction(reconstruction, rotation:R=R.identity(), 
                          origin:np.ndarray=np.array([0,0,0]), 
                          scaling:float=1.0
                          ) -> Dict[str, 
                                    Dict[str, 
                                         Dict[str, 
                                              Union[str, List[float]]]]]:
    
    """Updates the reconstruction dictionary with a new translation, 
    rotation and scaling.

    Arguments:
    ----------
        reconstruction (dict) :   reconstruction dictionary
        origin (np.ndarray)   :   translation vector
        rotation (Rotation)   :   rotation vector
        scaling (float)       :   scaling factor

    Returns:
    --------
        reconstruction (dict) :   updated reconstruction dictionary
    """
    new_reconstruction = copy.deepcopy(reconstruction)
            
    for i in new_reconstruction["shots"].keys():
        old_r = R.from_rotvec(reconstruction["shots"][i]["rotation"])
        new_r = (old_r * rotation.inv()).as_rotvec()
                                 
        new_reconstruction["shots"][i]["rotation"] = list(new_r)

        old_t = np.array(reconstruction["shots"][i]["translation"])
        new_t = scaling * old_t + old_r.apply(origin) 
        new_reconstruction["shots"][i]["translation"] = list(new_t)
    
    for i in new_reconstruction["points"].keys():
        old_coord = np.array(reconstruction["points"][i]["coordinates"])
        new_coord = rotation.apply(scaling * old_coord - origin)
        new_reconstruction["points"][i]["coordinates"] = list(new_coord)

    return new_reconstruction


