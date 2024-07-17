import pycolmap
import numpy as np
from pathlib import Path
from utils.metadata_utils import get_cam_names
from utils.multiview_utils import Camera, Calibration

# Function to generate a unique camera ID
def generate_unique_camera_id(model):
    existing_ids = set(model.cameras.keys())
    new_id = 1
    while new_id in existing_ids:
        new_id += 1
    return new_id

# Function to add a camera and image to the reconstruction
def add_camera_and_image(model, camera_data, cam_from_world, image_name):
    # Ensure the camera_id is set and unique
    if camera_data.camera_id == 4294967295:
        camera_data.camera_id = generate_unique_camera_id(model)
    
    # Add camera to the reconstruction
    model.add_camera(camera_data)
    
    # Ensure unique image_id
    image_id = max(model.images.keys(), default=0) + 1
    
    # Create and add image
    image = pycolmap.Image(cam_from_world=cam_from_world, )
    image.image_id = image_id
    image.camera_id = camera_data.camera_id
    image.name = image_name
    model.add_image(image)


def transform_points(points, camera):
    """
    Transform the points based on the camera pose.

    Parameters:
    points (ndarray): Array of points with shape (n, 3).
    camera (dict): Camera settings including 'eye' and 'up'.

    Returns:
    ndarray: Transformed points.
    """
    eye = np.array([camera['eye']['x'], camera['eye']['y'], camera['eye']['z']])
    up = np.array([camera['up']['x'], camera['up']['y'], camera['up']['z']])

    # Create a rotation matrix based on the camera orientation
    # For simplicity, we'll use the eye and up vectors to create a basic rotation matrix.
    # This example assumes a simple orthogonal transformation based on the up vector.
    
    # Normalize the up vector
    up = up / np.linalg.norm(up)

    # Calculate a right vector as orthogonal to the up vector
    right = np.cross([0, 0, 1], up)
    right = right / np.linalg.norm(right)

    # Calculate the forward vector
    forward = np.cross(up, right)
    forward = forward / np.linalg.norm(forward)

    # Create the rotation matrix
    rotation_matrix = np.vstack([right, up, forward]).T

    # Apply rotation and translation
    transformed_points = np.dot(points - eye, rotation_matrix)

    return transformed_points

# TODO: Verify setting of translation and rotation
def camera_calibs_from_colmap(images_path:Path, model:pycolmap.Reconstruction, omni_tag='360', save=True):
    cams = get_cam_names(images_path, omni_tag=omni_tag)
    for cam in cams:
        camera = Camera(cam)
        for id, image in model.images.items():
                if camera.name in image.name and omni_tag not in image.name:
                    world_t_camera = image.cam_from_world.inverse()
                    print(f"Setting Camera: {camera.name} position to: {world_t_camera} from image {image.name}")
                    camera.set_calib(
                        Calibration(
                                R = world_t_camera.rotation.matrix().T,
                                T = - world_t_camera.rotation.matrix().T @ world_t_camera.translation
                                ))
                    if save:
                        print("Saving Camera Calibration")
                        camera.save_calibration()
