import cv2
import copy

import numpy as np
import pyransac3d as pyrsc
import matplotlib.pyplot as plt
import ipywidgets as widgets

from typing import List, Tuple, Dict, Union, TypedDict, Optional
from scipy.spatial.transform import Rotation as R
from skspatial.objects import Points, Plane, Line, Vector, Point
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils.io_utils import write_json
from utils.coordinate_utils import update_reconstruction, point_in_polygon
from utils.multiview_utils import Camera, MultiviewVids


def rotation_matrix(v1:np.ndarray, v2:np.ndarray) -> np.ndarray:
    """
    Computes the rotation matrix between two vectors. Rotates
    the first vector to the second vector.

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        R: The rotation matrix.
    """

    # normalize the input vectors
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # check if the vectors are parallel
    if np.allclose(v1, v2) or np.allclose(v1, -v2):
        return np.eye(3)
    
    # compute the cross product and dot product
    cp = np.cross(v1, v2).astype(np.float64)
    dp = np.dot(v1, v2).astype(np.float64)
    
    # compute the skew-symmetric cross product matrix
    cp_list = [ [0., -cp[2], cp[1]],
                [cp[2], 0., -cp[0]],
                [-cp[1], cp[0], 0.]]
    cp_matrix = np.array(cp_list)
    
    # compute the rotation matrix
    R = np.eye(3) + cp_matrix + np.dot(cp_matrix, cp_matrix) * (1 / (1 + dp))
    
    return np.array(R, dtype=np.float64)


def perp(vector:np.ndarray) -> np.ndarray:
    """
    Computes a vector that is perpendicular to the input vector.

    Args:
        vector: The input vector.

    Returns:
        v: The perpendicular vector.
    """
    smallest = np.argmin(np.abs(vector))
    v = np.zeros(3)
    v[smallest] = 1.0
    return np.cross(vector, v)


def plot_3d_reconstruction(ax:Axes3D, reconstruction:dict, mvvids:MultiviewVids,  
                        ground_plane:Optional[Plane] = None, 
                        inliers: Optional[Points] = None, num_samples:int = 100
                        ) -> plt.Axes:
    """
    Plots the 3D reconstruction of the scene.

    Args:
        ax: The 3d axis object.
        reconstruction: The reconstruction dictionary.
        mvvids: The MultiviewVids object.
        ground_plane: The ground plane of the scene.
        inliers: The inliers of the ground plane.
        num_samples: The number of points to sample from the reconstruction.

    Returns:
        ax: The axis object.
    """

    # reconstruction points
    mvvids = copy.deepcopy(mvvids)
    plot_recon = []
    for i in reconstruction['points'].keys():
        plot_recon.append(reconstruction['points'][i]['coordinates'])

    # remove outliers from the reconstruction
    plot_recon_new = np.array(plot_recon)

    # detect outliers by computing the distance from the mean
    dist = np.linalg.norm(plot_recon - np.mean(plot_recon, axis=0), axis=1)
    plot_recon_new = plot_recon_new[dist < 2 * np.std(dist)]

    sample = np.random.choice(plot_recon_new.shape[0], 
                              num_samples, replace=False)
    Points(plot_recon_new[sample]).plot_3d(ax_3d=ax, c='b', s=1, marker='x', 
                                           alpha=0.4, 
                                           label='Reconstruction points')
    
    camera_centers = []
    for camera in mvvids.cams:
        camera.calib_from_reconstruction(reconstruction = reconstruction)
        camera_centers.append(camera.get_position())

    Points(camera_centers).plot_3d(ax_3d=ax, c='r', s=50, 
                                   label='Camera centers')

    # draw the rectangle on the ground
    bounding_box, _ = mvvids.get_bounding_box(reconstruction = reconstruction)
    
    rect_points = np.hstack([np.array(cv2.boxPoints(bounding_box)), 
                             np.zeros((4,1))])

    x, y, z = rect_points[:,0], rect_points[:,1], rect_points[:,2]
    verts = [(x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]), 
             (x[3], y[3], z[3])]
    poly = Poly3DCollection([verts], alpha=0.2)
    poly.set_facecolor('b')
    ax.add_collection3d(poly)

    if ground_plane is None:
        ground_plane = Plane(point = np.array([0,0,0]), 
                             normal = np.array([0,0,1]))
    else:
        ground_plane = Plane(point = ground_plane.point,
                                normal = ground_plane.normal)

    # draw the ground plane
    ground_plane.plot_3d(ax_3d=ax, alpha=0.5,label='Ground plane')
    Line(point = np.array([0,0,0]), 
         direction = ground_plane.normal).plot_3d(ax_3d=ax, c='r', t_2=200, 
                                                  label='Ground plane normal')
    
    if inliers is not None:
        inliers.plot_3d(ax_3d=ax, c='k', s=10, label='Inliers')
    

    # set limits of the plot to the limits of the cameras
    ax.set_xlim(Points(camera_centers)[:, 0].min() - 200, 
                Points(camera_centers)[:, 0].max() + 200)
    ax.set_ylim(Points(camera_centers)[:, 1].min() - 200, 
                Points(camera_centers)[:, 1].max() + 200)
    ax.set_zlim(0, Points(camera_centers)[:, 2].max() + 200)

    # set aspect ratio to be equal
    ax.set_aspect('equal')
    # add x_label
    ax.set_xlabel('X')
    # add y_label
    ax.set_ylabel('Y')
    # add z_label
    ax.set_zlabel('Z')

    return ax


def first_frame_selector(cameras:List[Camera], frame_ids:List[int],
                         base_frames:dict, json_path:str) -> None:
    """
    Creates a widget to select the first frame of the video.

    Args:
        cameras: The list of cameras.
        base_frames: The dictionary containing the extracted frames.
        json_path: The path to the json file containing the first frames.

    Returns:
        None
    """

    max_frame = min([len(base_frames[cam.name]) for cam in cameras])

    img_1_slider = widgets.IntSlider(
        value=0,
        min=0,
        max = max_frame // 1 - 1,
        step=1,
        description='Image left:',
        disabled=False
    )

    img2_slider = widgets.IntSlider(
        value=0,
        min=0,
        max = max_frame // 1 - 1,
        step=1,
        description='Image right:',
        disabled=False
    )
    selectable_cams = [cam.name for cam in cameras]
    selectable_cams.remove(cameras[0].name)
    cam_select = widgets.Dropdown(
        options=selectable_cams,
        value=cameras[1].name,
        description='Camera:',
        disabled=False,
    )

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)
    
    axs[0].imshow(base_frames[cameras[0].name][0])

    starting_frames = {}

    def reset_img2_slider(*args):
        img2_slider.value = 0

    cam_select.observe(reset_img2_slider, names='value')

    @widgets.interact(cam=cam_select, img1=img_1_slider, img2 = img2_slider)
    def update_imgs(cam=cameras[1].name, img1=0, img2=0):
        axs[0].clear()
        axs[1].clear()
        axs[0].imshow(base_frames[cameras[0].name][img1])
        axs[1].imshow(base_frames[cam][img2])
        axs[0].set_title('Frame {} for camera {}'.format(frame_ids[img1], 
                                                         cameras[0].name))
        axs[1].set_title('Frame {} for camera {}'.format(frame_ids[img2], 
                                                         cam))
        starting_frames[cameras[0].name] = int(frame_ids[img1])
        starting_frames[cam] = int(frame_ids[img2])
        
        # save the first frames dictionary
        write_json(json_path, starting_frames)
        plt.show()


class ScaleInfo(TypedDict):
    id: str
    point1: Tuple[float, float]
    point2: Tuple[float, float]
    distance: float


def get_scale(scale_info:ScaleInfo, mvvids:MultiviewVids, 
              ground_plane:Plane) -> float:
    """
    Computes the scale of the reconstruction based on the scale information.

    Args:
        scale_info: The dictionary containing the scale information.
                        Camera id, point1, point2, distance.

        mvvids: The MultiviewVids object.   

        ground_plane: The ground plane.

    Returns:
        The scale of the reconstruction.    
    """
    id = int(scale_info['id'])
    scale_cam = mvvids.cams[id - 1]
    point1 = scale_info['point1']
    point1 = np.array([[point1[0],point1[1], 1.]])
    point2 = scale_info['point2']
    point2 = np.array([[point2[0],point2[1], 1.]])
    distance = scale_info['distance']
   
    point1 = ground_plane.intersect_line(
        Line.from_points(
            Point(
                scale_cam.convert_to_world_frame(
                    scale_cam.from_image(point1)).reshape(3,)), 
                    Point(scale_cam.get_position())))
    point2 = ground_plane.intersect_line(
        Line.from_points(
            Point(
                scale_cam.convert_to_world_frame(
                    scale_cam.from_image(point2)).reshape(3,)),
                    Point(scale_cam.get_position())))
    
    scale = distance / np.linalg.norm(point1 - point2)

    return float(scale)


def align_ground(mvvids:MultiviewVids, reconstruction:dict, scale_info:ScaleInfo
                 ) -> Tuple[np.ndarray, R, float, Plane, Points, pyrsc.Plane]:
    """Aligns the ground plane of the reconstruction
    based on a region containing reconstruction keypoints
    that lie on the ground plane.
    
        Args:
            mvvids (MultiviewVids): MultiviewVids object
            reconstruction (dict): reconstruction dictionary
        
        Returns:
            Tuple[np.ndarray, np.ndarray, Plane, Points]: 
                R: rotation matrix
                t: translation vector
                ground_plane: ground plane final guess
                translated_gpp: ground plane points 
                pyrsc_plane: ground plane initial guess
                scale: scale factor for the reconstruction


    """
    reconstruction = copy.deepcopy(reconstruction)
    mvvids = copy.deepcopy(mvvids)
    ground_plane_points = []
    ground_plane_ids = []
    plot_recon = []

    for i in reconstruction['points'].keys():
        coordinate = reconstruction['points'][i]['coordinates']
        plot_recon.append(coordinate)
        for camera in mvvids.cams:
            if camera.calibration.ROI == []:
                continue
            projected_point, behind = camera.convert_to_camera_frame(
                    np.array([coordinate]))
            if behind:
                continue
            if point_in_polygon(projected_point[0], camera.calibration.ROI):
                ground_plane_points.append(coordinate)
                ground_plane_ids.append(i)
                break


    # remove outliers from the reconstruction
    plot_recon = np.array(plot_recon)
    ground_plane_points = np.array(ground_plane_points)

    # detect outliers by computing the distance from the mean
    dist = np.linalg.norm(plot_recon - np.mean(plot_recon, axis=0), axis=1)
    gpp_dist = np.linalg.norm(ground_plane_points 
                              - np.mean(ground_plane_points, axis=0), axis=1)
    ground_plane_points = ground_plane_points[gpp_dist < 2 * np.std(dist)]

    pyrsc_plane = pyrsc.Plane()
    pyrsc_plane.fit(np.array(ground_plane_points), thresh=0.05)
    pyrsc_normal = np.array(pyrsc_plane.equation[:3])
    point_on_plane  = np.array([0., 0., 
                                - pyrsc_plane.equation[3] 
                                / pyrsc_plane.equation[2]])

    # rotate normal onto the z-axis
    rotation = R.from_matrix(rotation_matrix(pyrsc_normal, np.array([0, 0, 1])))
    origin = np.array(point_on_plane)

    temp_reconstruction = update_reconstruction(copy.deepcopy(reconstruction), 
                                                rotation=rotation, 
                                                origin = origin)

    z_list = []
    for i in temp_reconstruction['points'].keys():
        z_list.append(temp_reconstruction['points'][i]['coordinates'][2])

    # remove outliers top and bottom 1%
    z_list = sorted(z_list)
    z_list = z_list[int(len(z_list)*0.02):int(len(z_list)*0.98)]
    z_median = np.median(z_list)

    # ensure that the z-axis points up
    if z_median < 0:
        rot = R.from_matrix(np.array([[1., 0., 0.], 
                                      [0., 1., 0.], 
                                      [0., 0., -1.]]))
        rotation = rot * rotation
        temp_reconstruction = update_reconstruction(
                                copy.deepcopy(reconstruction), 
                                rotation = rotation, 
                                origin = origin)
        
    translated_gpp = rotation.apply(
        np.array(copy.deepcopy(ground_plane_points)).astype(np.float64) 
        - origin)
    
    ground_plane = Plane.best_fit(Points(translated_gpp[pyrsc_plane.inliers]))

    # calculate new alignment
    rotation = (R.from_matrix(rotation_matrix(ground_plane.normal, 
                                             np.array([0, 0, 1]))) 
                                             * rotation)

    _, roi_array = mvvids.get_bounding_box(reconstruction = temp_reconstruction, 
                                           ground_plane=ground_plane)
    
    rotated_centroid = rotation.inv().apply(Points(roi_array).centroid())
    origin = np.array(origin + rotated_centroid).astype(np.float64)
    translated_gpp = rotation.apply(np.array(copy.deepcopy(ground_plane_points)) 
                                    - origin)
    ground_plane = Plane.best_fit(Points(translated_gpp[pyrsc_plane.inliers]))

    scale = get_scale(scale_info, mvvids, ground_plane)
    origin = np.array(origin * scale)
    translated_gpp = Points(translated_gpp * scale)
    ground_plane = Plane.best_fit(Points(translated_gpp[pyrsc_plane.inliers]))

    return origin, rotation, scale, ground_plane, translated_gpp, pyrsc_plane


def calculate_reprojection_error(mvv:MultiviewVids, reconstruction:dict,
                                 img: np.ndarray, 
                                 checkerboard_size:tuple = (9, 6),
                                 checkerboard_square_size:float = 32.0
                                 ) -> Optional[float]:
    """Calculates the reprojection error of a reconstruction
    based on the distance between the projected points and the
    keypoints in the image.
    
        Args:
            mvv (MultiviewVids): MultiviewVids object
            reconstruction (dict): reconstruction dictionary
            img (np.ndarray): image array
            checkerboard_size (tuple): size of the checkerboard (rows, columns)
            checkerboard_square_size (float): size of each square in the 
                                                checkerboard
        
        Returns:
            float: reprojection error

    """
    mvv = copy.deepcopy(mvv)
    reconstruction = copy.deepcopy(reconstruction)

    # Find checkerboard corners in the image
    ret, corners = cv2.findChessboardCorners(img, checkerboard_size)

    if ret:
        # Calculate object points for the checkerboard
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), 
                        np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                               0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= checkerboard_square_size

        # Refine corner positions
        corners2 = cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 
                                    corners, (11, 11), (-1, -1), 
                                    (cv2.TERM_CRITERIA_EPS 
                                     + cv2.TERM_CRITERIA_MAX_ITER, 
                                     30, 0.001))

        # Project object points to image plane
        rvec, tvec = reconstruction['R'], reconstruction['T']
        K = reconstruction['K']
        dist = reconstruction['D']
        imgpoints, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)

        # Calculate reprojection error
        error = np.mean(np.linalg.norm(corners2 - imgpoints.squeeze(), axis=1))

        return error
    else:
        return None