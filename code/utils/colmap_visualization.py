from typing import Optional

from utils.multiview_utils import MultiviewVids
from utils.metadata_utils import get_config_dict
from hloc.utils.viz_3d import plot_camera
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import copy
import numpy as np
import pycolmap
from utils.coordinate_utils import project_to_ground_plane_cv2
import ipywidgets as widgets
import cv2

from hloc.visualization import plot_images, read_image
from hloc.utils import viz_3d

from utils.log_utils import log
from utils.multiview_utils import MultiviewVids
from utils.metadata_utils import get_config_dict
from utils.colmap_utils import camera_calibs_from_colmap

from utils.colmap_utils import add_camera_and_image, camera_calibs_from_colmap

# Define necessary functions for ray-plane intersection
def ray_plane_intersection(ray_origin, ray_direction, plane_normal, plane_point):
    d = np.dot(plane_normal, plane_point)
    t = (d - np.dot(plane_normal, ray_origin)) / np.dot(plane_normal, ray_direction)
    return ray_origin + t * ray_direction

def image_to_world_coordinates(cam, roi_points, z_plane=0):
    intrinsic = cam.calibration.intrinsic  # Expecting a 3x3 numpy array
    rotation = cam.rotation_matrix()  # Rotation matrix of the camera
    translation = cam.translation_vector()  # Translation vector of the camera
    
    plane_normal = np.array([0, 0, 1])
    plane_point = np.array([0, 0, z_plane])

    world_points = []
    for point in roi_points:
        x, y = point
        # Convert image coordinates to normalized camera coordinates
        normalized_coords = np.linalg.inv(intrinsic) @ np.array([x, y, 1])
        # Convert normalized camera coordinates to world coordinates
        ray_direction = rotation @ normalized_coords
        ray_origin = translation
        # Find the intersection of the ray with the plane z=0
        world_point = ray_plane_intersection(ray_origin, ray_direction, plane_normal, plane_point)
        world_points.append(world_point)
    
    return np.array(world_points)


def plot_reconstruction_hloc(
    fig: go.Figure,
    rec: pycolmap.Reconstruction,
    max_reproj_error: float = 6.0,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    min_track_length: int = 2,
    points: bool = True,
    cameras: bool = True,
    points_rgb: bool = True,
    cs: float = 1.0,
    downsample_factor: int = 1,
):
    # Filter outliers
    bbs = rec.compute_bounding_box(0.01, 0.99)
    # Filter points, use original reproj error here
    p3Ds = [
        p3D
        for _, p3D in rec.points3D.items()
        if (
            (p3D.xyz >= bbs[0]).all()
            and (p3D.xyz <= bbs[1]).all()
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]

    p3Ds = p3Ds[::downsample_factor]
    
    xyzs = [p3D.xyz for p3D in p3Ds]
    if points_rgb:
        pcolor = [p3D.color for p3D in p3Ds]
    else:
        pcolor = color
    if points:
        viz_3d.plot_points(fig, np.array(xyzs), color=pcolor, ps=1, name=name)
    if cameras:
        viz_3d.plot_cameras(fig, rec, color=color, legendgroup=name, size=cs)
        
def plot_reconstruction(model, save_path=None):

    """plots a pycolmap reconstruction and adds regions of interest for the cameras
    """
    config = get_config_dict()
    fig = viz_3d.init_figure()
    
    plot_reconstruction_hloc(
    fig, model, color="rgba(255,0,0,0.5)", name="mapping", points_rgb=True, downsample_factor=10
    )

    mvvids = MultiviewVids(newest=False, config=config)
    # mvvids.cams = camera_calibs_from_colmap(images, model, save=False)

    bbx = model.compute_bounding_box(0.01, 0.99)

    log.debug(f"Bounding box: {bbx}")
    
    # Add a static plane at z=0
    plane = go.Mesh3d(
        x=[bbx[0][0], bbx[1][0], bbx[1][0], bbx[0][0]],
        y=[bbx[0][1], bbx[0][1], bbx[1][1], bbx[1][1]],
        z=[0, 0, 0, 0],
        color='lightblue',
        opacity=0.5,
        name='Plane at z=0',
        showscale=False
    )

    fig.add_trace(plane)

    # Process the cameras and plot the ROI intersection with the plane z=0
    for cam in mvvids.cams:
        roi_image_coords = cam.calibration.ROI
        
        if roi_image_coords is None:
            log.warning(f'No ROI found for camera {cam.name}')
            continue

        roi_camera_coords = cam.from_image(roi_image_coords)
        roi_world_coords = cam.convert_to_world_frame(roi_camera_coords)

        # Get the camera center in world coordinates
        camera_center = cam.get_position()

        plane_normal = np.array([0, 0, 1])
        plane_point = np.array([0, 0, 0])

        for point in roi_world_coords:
            ray_direction = point - camera_center
            intersection_point = ray_plane_intersection(camera_center, ray_direction, plane_normal, plane_point)
            
            # Plot the extended ray
            line = go.Scatter3d(
                x=[camera_center[0], intersection_point[0]],
                y=[camera_center[1], intersection_point[1]],
                z=[camera_center[2], intersection_point[2]],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f'ROI {cam.name}'
            )
            fig.add_trace(line)

    if save_path is not None:
        fig.write_html(save_path)
    
    return fig



def plot_ground_plane_proj(model, exclude:list = []):
    """
    Plots the groundplane projection for the set of static cameras in the reconstruction
    """
    # Initialize configuration and multiview video handler
    config = get_config_dict()
    mvvids = MultiviewVids(newest=False, config=config)
    # cameras = camera_calibs_from_colmap(images, temp_model, save=False)
    # mvvids.cams = [camera for camera in cameras if camera.name not in exclude]

    # Extract frames
    max_frame = np.min([10, mvvids.get_max_frame_id() - 1])
    step = 2
    frame_ids = list(np.arange(0, max_frame, step))
    base_frames = mvvids.extract_mv(frame_ids, undistort=True)

    # Setup plotly figure for 2D image projection
    fig_img = go.Figure()

    alphas = [1.0, 0.7, 0.5, 0.3]
    rect, _ = mvvids.get_bounding_box()
    output_img_size = (1920, 1080)
    input_img_size = (1080, 1920)

    for cam in mvvids.cams:
        img = copy.deepcopy(base_frames[cam.name][0])
        img = cv2.resize(img, (input_img_size[1], input_img_size[0]))
        H = cam.get_ground_plane_homography(input_img_size=input_img_size, output_img_size=output_img_size, bounding_box=rect, padding_percent=0)
        new_img = project_to_ground_plane_cv2(img, H, output_img_size)

        fig_img.add_trace(go.Image(z=new_img, opacity=alphas[int(cam.calibration.view_id) - 1]))


    return fig_img

def plot_histogram(model):
    """
    Plots the distribution of z-coordinates for a given pycolmap reconstruction
    """
    points = np.array([point.xyz for point in model.points3D.values()])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=points[:, 2], name='Z-Value Distribution'))

    return fig