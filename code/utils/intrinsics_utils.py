"""
How to acquire the video of the checkerboard:
1) Make sure the checkerboard is completely flat.
2) Always keep the checkerboard at an angle w.r.t the camera plane.
   Images of checkerbaords whose planes are parallel to the camera plane 
   are not useful.
3) Keep the checkerboard close to the camera. The area of the checkerboard 
    should intuitively be half of the area of the entire image.
4) Cover the corners of the image as well. It is fine if the checkerboard 
    goes outside the image; the algorithm will discard these automatically.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import datetime
import sys
import multiprocessing
import imageio
from itertools import repeat

from .io_utils import (load_json, write_json, 
                            rgb2gray, rmdir, mkdir, find_images)
from .log_utils import log

(cv2_major, cv2_minor, _) = cv2.__version__.split(".")
if int(cv2_major)<4:
    raise ImportError("Opencv version 4+ required!")

#----------------------------------------------------------------------------
# Created By  : Leonardo Citraro leonardo.citraro@epfl.ch
# Date: 2020
# ---------------------------------------------------------------------------

def distortion_function(points_norm: np.ndarray, 
                        dist: np.ndarray) -> np.ndarray:
    """
    Standard (OpenCV convention) distortion function
    
    Parameters
    ----------
    points_norm : numpy.ndarray (N,2)
        undistorted image points in normalized image coordinates.
        In other words, 3D object points transformed with [R,t]
    dist: list or numpy.ndarray (5,)
        distortion coefficients
    
    Return
    ------
    numpy.ndarray (N,2) distorted points
    """
    k_ = dist.reshape(5)

    x,y = points_norm[:,0], points_norm[:,1]
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    a1 = 2*x*y
    a2 = r2 + 2*x*x
    a3 = r2 + 2*y*y
    cdist = 1 + k_[0]*r2 + k_[1]*r4 + k_[4]*r6
    xd0 = x*cdist + k_[2]*a1 + k_[3]*a2
    yd0 = y*cdist + k_[2]*a3 + k_[3]*a1 

    return np.vstack([xd0, yd0]).T

def is_distortion_function_monotonic(dist: np.ndarray, 
                                     range: tuple=(0,1.5,100)) -> bool:
    """
    Checks if the distortion function is monotonic in the given range.
    The range is defined in the normalized image coordinate system.
    It starts at the principal point and moves away from it.

    Parameters
    ----------
    dist: list or numpy.ndarray (5,)
        distortion coefficients
    range : tuple (3,)
        range where the monotonicity must be checked

    Return
    ------
    bool: True if the distortion function is monotonic in the given range
    """
    
    x = np.linspace(*range)
    px = distortion_function(np.vstack([x,x]).T, dist)[:,0]

    return np.all((px[1:]-px[:-1])>0) 

def enforce_monotonic_distortion(dist, K, image_points, proj_undist_norm,
                                 range_constraint=(0, 1.4, 1000), 
                                 verbose=True):
    """
    Forces the distortion function to be monotonic in the given range.
    The range is defined in the normalized image coordinate system.
    It starts at the principal point and moves away from it.
    
    IMPORTANT: The monotonicity is enforced to the detriment of the accuracy 
    of the calibration. A large range will induce a higher error. 
    Before using this, try to sample more precise points on the corner of the 
    image first. If it is not enough, switch to the Rational Camera Model.  

    Parameters
    ----------
    dist: list or numpy.ndarray (5,)
        initial distortion coefficients
    K: numpy.ndarray (3,3)
        intrinsic matrix
    image_points_norm : numpy.ndarray (N,2)
        image points (distorted) in normalized image coordinates
    proj_undist_norm : numpy.ndarray (N,2)
        projected object points (undistorted) in normalized image coordinates
    range_constraint : tuple (3,)
        range where the monotonicity must be enforced
    Return
    ------
    numpy.ndarray (5, ) new distortion coefficients
    """
    from scipy.optimize import minimize

    def diffs(points, k):
        proj = distortion_function(points,k)
        return proj[1:,:]-proj[:-1,:]

    # these are the points we want to be monotonous after undistorting them
    x_constraint = np.linspace(*range_constraint)
    x_constraint = np.vstack([x_constraint, x_constraint]).T

    def f(k_new):
        undist_pts = cv2.undistortPoints(image_points, K, k_new).reshape(-1,2)
        cost = np.sum((undist_pts-proj_undist_norm)**2, axis=1).mean()
        return cost
    
    def ineq_constraint(k_new):
        return diffs(x_constraint, k_new)[:,0]

    con = {'type': 'ineq', 'fun': ineq_constraint, 'lb':0, 'ub':None}

    x0 = dist.copy().reshape(5,) + 0
    bounds=[(x-np.abs(1e-6), x+np.abs(1e-6)) 
            for x in x0[:-1]]+[(x0[-1]-1, x0[-1]+1)] # we only chnage k3
    
    res = minimize(f, x0, method='SLSQP', tol=1e-32, constraints=con, 
                   bounds=bounds, options={'ftol': 1e-32, 'eps': 1e-12, 
                                           'disp': verbose, 'maxiter':1000})
    log.debug(res)

    new_dist = res.x

    if not is_distortion_function_monotonic(new_dist, range_constraint):
        log.warning("Enforce monotonic distortion is unsuccessful, this does "
                    "not mean that the distortion parameters are bad.")

    return new_dist  

def probe_monotonicity(K, dist, newcameramtx, image_shape, N=100, M=100):
    
    # calculate the region in which to probe the monotonicity
    pts_undist = np.array([
        [0,0],
        [0,image_shape[0]],
        [image_shape[1],0],
        [image_shape[1], image_shape[0]]
    ])
    pts_norm = (pts_undist-newcameramtx[[0,1],[2,2]][None])/newcameramtx[[0,1],[0,1]][None]

    xmin, ymin = pts_norm.min(0)
    xmax, ymax = pts_norm.max(0)
    r_max = np.sqrt(xmax**2+ymax**2)

    # create points used to compute the sign after distortion
    alphas = np.linspace(0,np.pi/2, N//4+2)[1:-1]
    alphas = np.concatenate([alphas, alphas+np.pi/2, alphas+np.pi, alphas+np.pi*3/2])
    
    ds = r_max/M

    ptss = []
    sign = []
    for r in np.linspace(0, r_max, M):
        pts= np.vstack([r*np.cos(alphas), r*np.sin(alphas)]).T
        ptsp = np.vstack([(r+ds)*np.cos(alphas), (r+ds)*np.sin(alphas)]).T

        mask1 = np.logical_and(pts[:,0]>=xmin, pts[:,0]<xmax)
        mask2 = np.logical_and(pts[:,1]>=ymin, pts[:,1]<ymax)
        mask = np.logical_and(mask1, mask2)

        if np.all(mask==False):
            continue

        pts, ptsp = pts[mask],ptsp[mask]

        ptss.append((pts,ptsp))
        sign.append(np.sign(pts-ptsp))
        
    # distort the points
    grid, gridp = zip(*ptss)
    grid, gridp = np.vstack(grid), np.vstack(gridp)

    grid_ = np.hstack([grid, np.zeros((len(grid),1))])
    gridp_ = np.hstack([gridp, np.zeros((len(gridp),1))])

    proj1 = cv2.projectPoints(grid_, np.eye(3), np.zeros(3), np.eye(3), 
                              dist)[0].reshape(-1,2)
    proj2 = cv2.projectPoints(gridp_, np.eye(3), np.zeros(3), np.eye(3), 
                              dist)[0].reshape(-1,2)

    # probe 
    is_monotonic = np.sign(proj1-proj2)==np.vstack(sign)
    is_monotonic = np.logical_and(*is_monotonic.T)
    
    return grid, is_monotonic


def process_image(filename_image: str, inner_corners_height: int, 
                  inner_corners_width: int, debug: bool, 
                  debug_folder: str) -> np.ndarray:
    """
    Processes an image to find the chess board corners and returns the refined 
    image points.

    Args:
    - filename_image (str): The path to the image file.
    - inner_corners_height (int): The number of inner corners in the height 
                                    direction.
    - inner_corners_width (int): The number of inner corners in the width 
                                    direction.
    - debug (bool): Whether to save the debug image or not.
    - debug_folder (str): The path to the folder where the debug image will 
                                    be saved.

    Returns:
    - np.ndarray: The refined image points.

    Raises:
    - None

    Example usage:
    ```
    imgp = process_image('image.jpg', 6, 9, True, 'debug_folder')
    ```
    """
    log.debug("Processing image {} ...".format(filename_image))

    gray = rgb2gray(imageio.imread(filename_image))

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, 
                                             (inner_corners_height,
                                              inner_corners_width),
                                             cv2.CALIB_CB_ADAPTIVE_THRESH 
                                             + cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If found, add object points, image points (after refining them)
    if not ret:
        return None
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    imgp = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    if debug:
        gray = cv2.drawChessboardCorners(gray, 
                                         (inner_corners_height, 
                                          inner_corners_width), 
                                          imgp, ret)
        imageio.imsave(os.path.join(debug_folder, 
                                    os.path.basename(filename_image)), gray)

    return np.float32(imgp)
    

def compute_intrinsics(folder_images: str, output_folder:str, args) -> dict:

    """Computes the intrinsics for a camera given a folder of images of a checkerboard pattern.

    Arguments:
    -----------
        folder_images(str): folder containing the frames of the checkerboard pattern
        output_folder(str): folder where the data will be saved

    Returns:
    -----------
        intrinsics(dict): dictionary containing the intrinsics of the camera
    """

    # extract args 
    description = args.description
    inner_corners_height = args.inner_corners_height
    inner_corners_width = args.inner_corners_width
    square_sizes = args.square_sizes
    rational_model = args.rational_model
    alpha = args.alpha
    force_monotonicity = args.force_monotonicity
    monotonic_range = args.monotonic_range
    intrinsic_guess = args.intrinsic_guess
    fix_principal_point = args.fix_principal_point
    fix_aspect_ratio = args.fix_aspect_ratio
    zero_tangent_dist = args.zero_tangent_dist
    criteria_eps = args.criteria_eps
    threads = args.threads
    debug = args.debug
    load_keypoints = args.load_keypoints
    save_keypoints = args.save_keypoints
    fix_k1 = args.fix_k1
    fix_k2 = args.fix_k2
    fix_k3 = args.fix_k3
    fix_k4 = args.fix_k4
    fix_k5 = args.fix_k5
    fix_k6 = args.fix_k6
    
    debug_folder = os.path.join(output_folder, "debug")
    undistorted_folder = os.path.join(output_folder, "undistorted")

    # delete if exist
    rmdir(debug_folder)
    # rmdir(undistorted_folder)

    mkdir(undistorted_folder)
    if debug:
        mkdir(debug_folder)

    log.info("-" * 50)
    log.info("Input parameters")
    log.info("-" * 50)
    log.info(f"folder_images: {folder_images}")
    log.info(f"output_folder: {output_folder}")
    log.info(f"description: {description}")
    log.info(f"inner_corners_height: {inner_corners_height}")
    log.info(f"inner_corners_width: {inner_corners_width}")
    log.info(f"square_sizes: {square_sizes}")
    log.info(f"rational_model: {rational_model}")
    log.info(f"alpha: {alpha}")
    log.info(f"force_monotonicity: {force_monotonicity}")
    log.info(f"monotonic_range: {monotonic_range}")
    log.info(f"intrinsic_guess: {intrinsic_guess if len(intrinsic_guess) else False}")
    log.info(f"fix_principal_point: {fix_principal_point}")
    log.info(f"fix_aspect_ratio: {fix_aspect_ratio}")
    log.info(f"zero_tangent_dist: {zero_tangent_dist}")
    log.info(f"criteria_eps: {criteria_eps}")
    log.info(f"threads: {threads}")
    log.info(f"debug: {debug}")
    log.info("-" * 50)

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    if load_keypoints:
        keypoints = load_json(os.path.join(output_folder, "keypoints.json"))
        objpoints = np.float32(keypoints['objpoints'])
        imgpoints = np.float32(keypoints['imgpoints'])
    else:
        # prepare object points, like (0,0,0), (30,0,0), (60,0,0) ....
        # each square is 30x30mm
        # NB. the intrinsic parameters, rvec and distCoeffs do not depend 
        # upon the chessboard size, tvec does
        objp = np.zeros((inner_corners_height * inner_corners_width, 3), 
                        np.float32)
        objp[:,:2] = np.mgrid[0:inner_corners_height,
                              0:inner_corners_width].T.reshape(-1,2)
        objp[:,:2] *= square_sizes

        filename_images = find_images(folder_images, "*")
        if len(filename_images) == 0:
            log.warning("-" * 50)
            log.warning("WARNING! Unable to detect images in this folder!")
            sys.exit(0)
        if threads > 0:
            with multiprocessing.Pool(threads) as pool:
                res = pool.starmap(process_image, 
                                   zip(filename_images,
                                       repeat(inner_corners_height),
                                       repeat(inner_corners_width), 
                                       repeat(debug),
                                       repeat(debug_folder)))
        else:
            res = [process_image(f, inner_corners_height, inner_corners_width,
                                    debug, debug_folder) 
                                    for f in filename_images]

        nb_valid = sum(x is not None for x in res)
        log.info(f"Nb valid frames: {nb_valid} - therefore "
                 f"{nb_valid/len(res)*100:.2f}% of the calibration frames "
                 "contains valid chessboards")
        
        # 3d points in real world space
        objpoints = [objp.copy() for r in res if r is not None] 
        # 2d points in image plane
        imgpoints = [r.copy() for r in res if r is not None] 

        if save_keypoints:
            write_json(os.path.join(output_folder, "keypoints.json"), 
                       {'objpoints':np.float32(objpoints).tolist(), 
                        'imgpoints':np.float32(imgpoints).tolist()})

    image = imageio.imread(filename_images[0])
    image_shape = image.shape[:2]

    all_corners = np.vstack(imgpoints).squeeze()
    hist = np.histogram2d(all_corners[:,0], all_corners[:,1], 
                          bins=(np.arange(0, image_shape[1], 10), 
                                np.arange(0, image_shape[0], 10)),
                                density=True)[0]

    log.info(f"Saving to {os.path.join(output_folder, 'detected_2d_histogram.jpg')}")
    plt.figure()
    plt.imshow(np.array(hist).T.astype(float))
    plt.savefig(os.path.join(output_folder, "detected_2d_histogram.jpg"), 
                bbox_inches='tight')

    # visualize the keypoints
    plt.figure()
    plt.plot(*np.vstack(imgpoints).squeeze().transpose(1,0), 'g.')
    plt.grid()
    plt.xlim(0, image_shape[1])
    plt.ylim(image_shape[0], 0)
    plt.savefig(os.path.join(output_folder, "detected_keypoints.jpg"), 
                bbox_inches='tight')
    
    calib_flags = 0
    if rational_model:
        calib_flags += cv2.CALIB_RATIONAL_MODEL
    if fix_principal_point:
        calib_flags += cv2.CALIB_FIX_PRINCIPAL_POINT
    if fix_aspect_ratio:
        calib_flags += cv2.CALIB_FIX_ASPECT_RATIO
    if zero_tangent_dist:
        calib_flags += cv2.CALIB_ZERO_TANGENT_DIST
    if fix_k1:
        calib_flags += cv2.CALIB_FIX_K1
    if fix_k2:
        calib_flags += cv2.CALIB_FIX_K2
    if fix_k3:
        calib_flags += cv2.CALIB_FIX_K3
    if fix_k4:
        calib_flags += cv2.CALIB_FIX_K4
    if fix_k5:
        calib_flags += cv2.CALIB_FIX_K5
    if fix_k6:
        calib_flags += cv2.CALIB_FIX_K6        
        
    K_guess, dist_guess = None, None
    if len(intrinsic_guess):
        intrinsic_guess = load_json(intrinsic_guess)
        K_guess = np.array(intrinsic_guess['K'])
        dist_guess = np.array(intrinsic_guess['dist'])
        calib_flags += cv2.CALIB_USE_INTRINSIC_GUESS
        
        log.debug("K_guess:", K_guess)
        log.debug("dist_guess:", dist_guess)

    log.debug("Calibrating Camera...")
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                30, criteria_eps)
    
    iFixedPoint = inner_corners_height-1

    ret, mtx, distCoeffs, rvecs, tvecs, newObjPoints, \
    stdDeviationsIntrinsics, stdDeviationsExtrinsics, \
    stdDeviationsObjPoints, perViewErrors = cv2.calibrateCameraROExtended(
        objpoints, imgpoints, image_shape[::-1], iFixedPoint, K_guess, 
        dist_guess, flags=calib_flags, criteria=criteria)
    
    def reprojection_error(mtx, distCoeffs, rvecs, tvecs):
        # print reprojection error
        reproj_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                              mtx, distCoeffs)
            reproj_error += cv2.norm(imgpoints[i], 
                                     imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        reproj_error /= len(objpoints) 
        return reproj_error
    
    reproj_error = reprojection_error(mtx, distCoeffs, rvecs, tvecs)
    log.info(f"RMS Reprojection Error: {ret}, "
             f"Total Reprojection Error: {reproj_error}")
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distCoeffs, 
                                                      image_shape[::-1], alpha, 
                                                      image_shape[::-1], 
                                                      centerPrincipalPoint=True)
    
    grid_norm, is_monotonic = probe_monotonicity(mtx, distCoeffs, newcameramtx, 
                                                 image_shape, N=100, M=100)
    if not np.all(is_monotonic):
        log.warning("-" * 50)
        log.warning(" The distortion function is not monotonous "
                    "for alpha={:0.2f}!".format(alpha))
        log.warning(" To fix this we suggest sampling more precise points on "
                    "the corner of the image first.")
        log.warning(" If this is not enough, use the option Rational "
                    "Camera Model which more adpated to wider lenses.")
        log.warning("-" * 50)
    
    # visualise monotonicity
    plt.figure()
    plt.imshow(cv2.undistort(image, mtx, distCoeffs, None, newcameramtx))
    grid = (grid_norm * newcameramtx[[0,1],[0,1]][None] 
            + newcameramtx[[0,1],[2,2]][None])
    plt.plot(grid[is_monotonic, 0], grid[is_monotonic, 1], 
             '.g', label='monotonic', markersize=1.5)
    plt.plot(grid[~is_monotonic, 0], grid[~is_monotonic, 1], 
             '.r', label='not monotonic', markersize=1.5)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_folder, "monotonicity.jpg"), 
                bbox_inches='tight')
    
    proj_undist_norm = np.vstack([cv2.projectPoints(objpoints[i], rvecs[i], 
                                                    tvecs[i], np.eye(3), 
                                                    None)[0].reshape(-1,2)
                                     for i in range(len(rvecs))])
    
    if force_monotonicity:
        is_monotonic = is_distortion_function_monotonic(distCoeffs, 
                                            range=(0, monotonic_range, 1000))
        if is_monotonic:
            log.info("The distortion function is monotonic in the range "
                     "(0,{:0.2f})".format(monotonic_range))
        else:
            log.info("The distortion function is not monotonic in the range "
                     "(0,{:0.2f})".format(monotonic_range))

        if not is_monotonic:
            log.info("Trying to enforce monotonicity in the range "
                     "(0,{:.2f})".format(monotonic_range))

            image_points = np.vstack(imgpoints)
            distCoeffs = enforce_monotonic_distortion(
                distCoeffs, mtx, image_points, proj_undist_norm, 
                range_constraint=(0, monotonic_range, 1000))

            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, distCoeffs, image_shape[::-1], alpha, 
                image_shape[::-1], centerPrincipalPoint=False)
            
            rvecs_new, tvecs_new = [],[]

            for objp,imgp in zip(objpoints, imgpoints):
                _, rvec, tvec = cv2.solvePnP(objp, imgp, mtx, distCoeffs)
                rvecs_new.append(rvec)
                tvecs_new.append(tvec)

            reproj_error = reprojection_error(mtx, distCoeffs, 
                                              rvecs_new, tvecs_new)
            log.info(f"mono: RMS Reprojection Error: {ret}, "
                     f"Total Reprojection Error: {reproj_error}")

    d_json = dict({"date":current_datetime, 
                   "description":description,
                   "K":mtx.tolist(), 
                   "K_new":newcameramtx.tolist(), 
                   "dist":distCoeffs.ravel().tolist(),
                   "reproj_error":reproj_error, 
                   "image_shape":image_shape})

    # The code from this point on as the purpose of verifiying that the estimation went well.
    # images are undistorted using the compouted intrinsics
    
    # undistorting the images
    log.info("Saving undistorted images..")
    for i,filenames_image in enumerate(filename_images):
        if i % 100 != 0:
            continue
        print(f'Saving File: {filenames_image.split("/")[-1]} \n')
        img = imageio.imread(filenames_image)
        h, w = img.shape[:2]

        try:
            dst = cv2.undistort(img, mtx, distCoeffs, None, newcameramtx)
            
            # draw principal point
            dst = cv2.circle(dst, (int(mtx[0, 2]), int(mtx[1, 2])), 6, (255, 0, 0), -1)

            imageio.imsave(os.path.join(undistorted_folder, 
                                        os.path.basename(filenames_image)), dst)
        except:
            log.warning("Something went wrong while undistorting the images. "
                  "The distortion coefficients are probably not good. "
                  "You need to take a new set of calibration images.")
            #sys.exit(0)     
    
    for r, f in zip(res, filename_images):
        if r is None:
            log.warning(f"WARNING! Unable to detect chessboard in image {f}, "
                        "image will be removed")
            os.remove(f)
            
    return d_json