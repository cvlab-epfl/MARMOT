import colorsys
from pathlib import Path

import cv2
import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils.log_utils import log
from torchvision import transforms
from typing import Tuple, Optional, List, Dict, Union, Any


import torch

def is_in_frame(point:Tuple[float, float], frame_size: Tuple[int, int]) -> bool:
    """Check if a point is in the frame

    Args:
        point (Tuple[float, float]): Point to check
        frame_size (Tuple[int, int]): Size of the frame

    Returns:
        bool: True if point is in frame
    """
    is_in_top_left = point[0] > 0 and point[1] > 0
    is_in_bottom_right = point[0] < frame_size[0] and point[1] < frame_size[1]
    
    return is_in_top_left and is_in_bottom_right

def visualize_gt_cv2(input_img, gt_points=None, roi=None, pred_points=None):
    input_img = np.array(input_img, dtype = np.uint8)
    """
    Visualizes ground truth points and/or a region of interest (ROI) 
    on an input image.

    Args:
        input_img (np.ndarray): The input image to visualize.
        gt_points (Optional[List], optional): A list of ground truth points to 
                                    visualize on the image. Defaults to None.
        roi (Optional[np.ndarray], optional): A region of interest to 
                                    visualize on the image. Defaults to None.

    Returns:
        np.ndarray: The output image with the ground truth points 
                                    and/or ROI visualized.
    """
    
    # Convert input image to uint8 data type
    input_img = np.array(input_img, dtype=np.uint8)

    # Create a copy of the input image
    out_image = input_img.copy()
    
    # Visualize the ROI if provided
    if roi is not None:
        roi = np.array(roi.squeeze() * 255, dtype=np.uint8)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE)
        out_image = cv2.addWeighted(out_image, 0.8, roi, 0.4, 0)
    
    # Visualize the ground truth points if provided
    if gt_points is not None:
        for point in gt_points:
            # Check if the point is within the image frame
            if is_in_frame(point, out_image.shape[:2][::-1]):
                out_image = cv2.drawMarker(out_image, (int(point[0]), int(point[1])), [0,190,0], cv2.MARKER_SQUARE, markerSize=6, thickness=3) #MarkerType[, markerSize[, thickness[, line_type]]]]	)

    if pred_points is not None:
        # log.debug(gt_points)
        # log.debug("#######")
        for point in pred_points:
            if is_in_frame(point, out_image.shape[:2][::-1]):
                out_image = cv2.drawMarker(out_image, (int(point[0]), int(point[1])), [190,0,0], cv2.MARKER_CROSS, markerSize=6, thickness=3) #MarkerType[, markerSize[, thickness[, line_type]]]]	)
    

    return out_image

def visualize_hm_det(epoch_result_dicts: Dict, det: str, conf: Dict) -> List:
    """
    Visualizes heatmaps and detections on a sequence of frames.

    Args:
        epoch_result_dicts (Dict): A dictionary containing the results 
                                    of the epoch.
        det (str): The type of detection to visualize.
        conf (Dict): A dictionary containing configuration parameters.

    Returns:
        List: A list of visualization frames.
    """
    
    # Create an empty list to store the visualization frames
    visualization_frames = list()
    
    # Extract the frame ID from the detection string
    f_id = det.split("_")[1][0]

    # Iterate over each frame in the epoch result dictionary
    for frame_id in range(len(epoch_result_dicts[det])):
        
        # Extract the metric statistics for the current frame
        metrics = epoch_result_dicts["metric_stats"]

        # Determine the prediction scale based on the image plane configuration
        if conf["data_conf"]["image_plane"]:
            pred_scale = 8
        else:
            pred_scale = 4

        # Visualize the heatmaps and detections for the current frame
        if det.split("_")[0] == "framepred":
            v_id = int(det.split('_')[2][1])
            frame_flow = visualize_density_cv2(
                epoch_result_dicts[f"frame_image_{f_id}_v{v_id}"][frame_id], 
                epoch_result_dicts[det][frame_id], 
                det, 
                points=epoch_result_dicts[det+"_points"][frame_id], 
                count=0, 
                count_det=len(epoch_result_dicts[det+"_points"][frame_id]), 
                prec=metrics[f"precision_{det}"][frame_id], 
                rec=metrics[f"recall_{det}"][frame_id], 
                gt_points=np.array(epoch_result_dicts[
                    f"gt_points_image_\{f_id}_v{v_id}"][frame_id]).astype(int), 
                roi=epoch_result_dicts[f"roi_image_v{v_id}"][frame_id],
                pred_scale=8
                )
        else:
            frame_flow = visualize_density_cv2(
                epoch_result_dicts[f"frame_{f_id}"][frame_id], 
                epoch_result_dicts[det][frame_id], 
                det, 
                points=epoch_result_dicts[det+"_points"][frame_id], 
                count=0, 
                count_det=len(epoch_result_dicts[det+"_points"][frame_id]), 
                prec=metrics[f"precision_{det}"][frame_id], 
                rec=metrics[f"recall_{det}"][frame_id], 
                gt_points=epoch_result_dicts[f"gt_points_{f_id}"][frame_id], 
                roi=epoch_result_dicts["roi"][frame_id],
                pred_scale=pred_scale
                )

        # Add the visualization frame to the list
        visualization_frames.append(frame_flow)

    # Return the list of visualization frames
    return visualization_frames

def visualize_motion(epoch_result_dicts: Dict, motion: str, conf: Dict) -> List:
    """
    Visualizes motion between two frames.

    Args:
        epoch_result_dicts (Dict): A dictionary containing the results 
                                        of the epoch.
        motion (str): The motion to visualize.
        conf (Dict): A dictionary containing configuration parameters.

    Returns:
        List: A list of visualization frames.
    """
    
    # Create an empty list to store the visualization frames
    visualization_frames = list()
    
    # Extract the start and end frame IDs from the motion string
    start_id = motion.split("_")[1][0]
    end_id = motion.split("_")[2][0]

    # Extract the motion direction from the motion string
    motion_direction = motion.split("_")[-1][-1]

    # Construct the detection and recognition strings based 
    # on the start and end frame IDs
    det = f"det_{start_id}{motion_direction}"
    rec = f"rec_{motion.split('_')[2]}"

    # Generate a log message indicating the motion being visualized
    log.debug(f"Generating visualization for motion {motion} based"
               "on {det} and producing {rec}")

    # Iterate over each frame in the motion sequence
    for frame_id in range(len(epoch_result_dicts[motion])):

        # Extract the metric statistics for the current frame
        metrics = epoch_result_dicts["metric_stats"]

        # Generate a list of flow pairs for the current frame
        flow_pairs =  generate_motion_tuple(
            epoch_result_dicts[f"gt_points_{start_id}"][frame_id],
            epoch_result_dicts[f"person_id_{start_id}"][frame_id],
            epoch_result_dicts[f"gt_points_{end_id}"][frame_id],
            epoch_result_dicts[f"person_id_{end_id}"][frame_id]
            ) 

        # Visualize the motion for the current frame
        if conf["model_conf"]["flow_model"] == "cont":
            # Visualize offset
            frame_flow = generate_det_map_with_arrow(
            epoch_result_dicts[f"frame_{start_id}"][frame_id], 
            epoch_result_dicts[det][frame_id], 
            epoch_result_dicts[rec+"_points"][frame_id], 
            epoch_result_dicts[motion][frame_id],
            flow_pairs,
            motion,
            legend_det=f"T{start_id}", 
            legend_rec=f"T{end_id}")
        elif (conf["model_conf"]["flow_model"] == "prob" 
              or conf["model_conf"]["flow_model"] == "org"):
            # Visualize flow map
            frame_flow = generate_flow_map_with_arrow(
            epoch_result_dicts[f"frame_{start_id}"][frame_id], 
            epoch_result_dicts[det][frame_id], 
            epoch_result_dicts[rec+"_points"][frame_id], 
            epoch_result_dicts[motion][frame_id],
            flow_pairs,
            motion,
            legend_det=f"T{start_id}", 
            legend_rec=f"T{end_id}")
        else:
            # Raise an error if the flow model is not recognized
            raise NotImplementedError("Visualization for flow is "
                                      "not implemented")
        
        # Add the visualization frame to the list
        visualization_frames.append(frame_flow)

    # Return the list of visualization frames
    return visualization_frames


def visualize_bbox_det(frame:np.ndarray, pred_bbox:np.ndarray, 
                       gt_bbox:Optional[np.ndarray] = None, 
                       anchor_points:Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualizes bounding boxes on an input image.

    Args:
        frame (np.ndarray): The input image to visualize.
        pred_bbox (np.ndarray): The predicted bounding box to visualize 
                                    on the image.
        gt_bbox (Optional[np.ndarray], optional): The ground truth bounding box 
                                    to visualize on the image. Defaults to None.
        anchor_points (Optional[np.ndarray], optional): The anchor points to 
                                    use for visualizing the bounding box. 
                                    Defaults to None.

    Returns:
        np.ndarray: The output image with the bounding boxes visualized.
    """
    
    # Create a copy of the input image
    frame = np.copy(frame)
    
    # Convert the input image to a contiguous array
    frame = np.ascontiguousarray(frame)

    # Visualize the predicted bounding box on the image
    frame = visualize_bounding_box(pred_bbox, frame, anchor_points, 
                                   thickness=2, color="rnd")
    
    # Visualize the ground truth bounding box on the image if provided
    if gt_bbox is not None:
        frame = visualize_bounding_box(gt_bbox, frame, thickness=1, 
                                       color= (1, 1, 1))
    
    # Return the output image
    return frame
    

def visualize_bounding_box(bbox_list: np.ndarray, frame: np.ndarray, 
                           anchor_points: Optional[np.ndarray] = None, 
                           thickness: int = 5, 
                           color: Union[str, Tuple[float, float, float]] = "rnd"
                           ) -> np.ndarray:
    """
    Visualizes bounding boxes on an input image.

    Args:
        bbox_list (np.ndarray): The list of bounding boxes to visualize 
                                    on the image.
        frame (np.ndarray): The input image to visualize.
        anchor_points (Optional[np.ndarray], optional): The anchor points to 
                    use for visualizing the bounding boxes. Defaults to None.
        thickness (int, optional): The thickness of the bounding box 
                                        and anchor point markers. Defaults to 5.
        color (Union[str, Tuple[float, float, float]], optional): The color of 
                the bounding box and anchor point markers. Defaults to "rnd".

    Returns:
        np.ndarray: The output image with the bounding boxes visualized.
    """
    
    # Set the initial color value
    color_u = color

    # Visualize the bounding boxes on the image
    if anchor_points is None:
        for bb in bbox_list:
            if color == "rnd":
                color_u = list(np.random.random(size=3))

            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), 
                                  color_u, thickness)
    else:
        for bb, ap in zip(bbox_list,anchor_points):
            if color == "rnd":
                color_u = list(np.random.random(size=3))

            frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), 
                                  color_u, thickness)
            frame = cv2.drawMarker(frame, tuple(ap.astype(int)), 
                                   color_u, cv2.MARKER_DIAMOND, 
                                   markerSize=5, thickness=thickness)
        
    # Return the output image
    return frame


def visualize_multi_hm(epoch_result_dicts: Dict[str, Any], 
                       conf: Dict[str, Any], type: str) -> List[np.ndarray]:
    """
    Visualizes the heatmaps for multiple frames.

    Args:
        epoch_result_dicts (Dict[str, Any]): The dictionary containing the 
                                                results for each frame.
        conf (Dict[str, Any]): The configuration dictionary.
        type (str): The type of detection to visualize.

    Returns:
        List[np.ndarray]: The list of output images with the 
                                heatmaps visualized.
    """
    
    # Initialize the list of output images
    visualization_frames = list()
    
    # Loop through each frame in the epoch result dictionary
    for frame_id in range(len(epoch_result_dicts["frame"])):
        
        # Get the results and metrics for the current frame
        results = epoch_result_dicts["processed_results"]
        metrics = epoch_result_dicts["metric_stats"]

        # Visualize the ground plane heatmap
        frame_center = visualize_density_cv2(
            epoch_result_dicts["frame_groundplane"][frame_id], 
            epoch_result_dicts[f"hm_{type}_ground_solo"][frame_id], 
            f"{type} Det. ground", 
            points=results[f"pred_point_{type}_ground"][frame_id], 
            gt_points=epoch_result_dicts["gt_points"][frame_id], 
            roi=epoch_result_dicts["roi"][frame_id]
            )
        
        # Visualize the head plane heatmap
        frame_flow = visualize_density_cv2(
            epoch_result_dicts["frame_headplane"][frame_id], 
            epoch_result_dicts[f"hm_{type}_head_solo"][frame_id], 
            f"{type} Det. head", 
            points=results[f"pred_point_{type}_head"][frame_id],
            gt_points=epoch_result_dicts["gt_points"][frame_id], 
            roi=epoch_result_dicts["roi"][frame_id]
            )

        # Combine the two heatmaps into a single output image
        visualization_frames.append(combine_two_frame_cv2(frame_center, 
                                                          frame_flow))

    # Return the list of output images
    return visualization_frames


def visualize_track(epoch_result_dicts: Dict[str, Any], conf: Dict[str, Any], 
                    left: str = "center", 
                    right: str = "flow") -> List[np.ndarray]:
    """
    Visualizes the tracking results for the left and right cameras.

    Args:
        epoch_result_dicts (Dict[str, Any]): The dictionary containing the 
                                                results for each frame.
        conf (Dict[str, Any]): The configuration dictionary.
        left (str, optional): The type of tracking to visualize for the 
                                    left camera. Defaults to "center".
        right (str, optional): The type of tracking to visualize for the 
                                    right camera. Defaults to "flow".

    Returns:
        List[np.ndarray]: The list of output images with the tracking 
                                results visualized.
    """
    
    # Determine the prediction scale based on the image plane configuration
    if conf["data_conf"]["image_plane"]:
        pred_scale = 8
    else:
        pred_scale = 4

    # Generate the tracking visualization frames for the left and right cameras
    frames_track_left = make_tracking_vis(
        epoch_result_dicts["frame_groundplane"], 
        epoch_result_dicts["tracking"][left], f"{left} Tracking", 
        epoch_result_dicts["roi"], pred_scale)
    frames_track_right = make_tracking_vis(
        epoch_result_dicts["frame_groundplane"], 
        epoch_result_dicts["tracking"][right], 
        f"{right} Tracking", 
        epoch_result_dicts["roi"], 
        pred_scale)
        
    # Combine the tracking visualization frames for the left and right 
    # cameras into a single output image
    visualization_tracking_frames = list()
    for frame_id in range(len(frames_track_left)):
        visualization_tracking_frames.append(
            combine_two_frame_cv2(frames_track_left[frame_id], 
                                  frames_track_right[frame_id]))

    # Return the list of output images
    return visualization_tracking_frames


def combine_two_frame_cv2(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Combines two input frames into a single output frame.

    Args:
        frame1 (np.ndarray): The first input frame.
        frame2 (np.ndarray): The second input frame.

    Returns:
        np.ndarray: The output frame with the two input frames combined.
    """
    
    # Add a white border to the right of the first frame and to the 
    # bottom of the second frame
    frame1 = cv2.copyMakeBorder(frame1, 0, 0, 0, 8, 
                                cv2.BORDER_CONSTANT, value=[255,255,255])
    frame2 = cv2.copyMakeBorder(frame2, 0, 0, 8, 0, 
                                cv2.BORDER_CONSTANT, value=[255,255,255])

    # Concatenate the two frames horizontally to create the output frame
    combined_frame = np.concatenate((frame1, frame2), axis=1)
    
    # Return the output frame
    return combined_frame



def generate_det_map_with_arrow(input_img, heatmap_pred, det_point_rec, 
                                flow_hm, flow_pairs, title, scaleup=10, 
                                legend_det="T", legend_rec="T-1"):
    """
    Generates a detection map with arrows overlaid on top of an input image.

    Args:
        input_img (np.ndarray): The input image.
        heatmap_pred (np.ndarray): The predicted heatmap.
        det_point_rec (List[Tuple[int, int]]): The list of reconstructed 
                                                detection points.
        flow_hm (np.ndarray): The flow heatmap.
        flow_pairs (List[Tuple[Tuple[int, int], Tuple[int, int]]]): The list of 
                                                flow pairs.
        title (str): The title of the output image.
        scaleup (int, optional): The scale factor for the output image. 
                                                Defaults to 10.
        legend_det (str, optional): The legend for the detection. 
                                                Defaults to "T".
        legend_rec (str, optional): The legend for the reconstruction. 
                                                Defaults to "T-1".

    Returns:
        np.ndarray: The output image with the detection map and arrows 
                        overlaid on top of the input image.
    """
    
    # Convert the flow heatmap to a numpy array
    flow_hm = flow_hm.numpy()
    
    # Convert the input image and predicted heatmap to numpy arrays
    input_img = np.array(input_img * 255, dtype = np.uint8)
    density_pred = np.array(heatmap_pred * 255, dtype = np.uint8).squeeze()        
    density_pred = cv2.normalize(density_pred, None, alpha=0, 
                                 beta=255, norm_type=cv2.NORM_MINMAX)
    
    # Resize the predicted heatmap if necessary
    if scaleup != 1:
        density_pred = cv2.resize(density_pred, 
                                  tuple([c*scaleup 
                                         for c 
                                         in density_pred.shape[:2]][::-1]))

    # Apply a color map to the predicted heatmap
    heatmap_img = cv2.applyColorMap(density_pred, cv2.COLORMAP_RAINBOW)
    
    # Resize the input image if necessary
    if input_img.shape[:2] != heatmap_img.shape[:2]:
        input_img = cv2.resize(input_img, heatmap_img.shape[:2][::-1])

    # Combine the input image and predicted heatmap
    out_image = cv2.addWeighted(input_img, 0.8, heatmap_img, 0.5, 0)
    
    # Draw arrows for all points on a grid representing prediction motion
    # Color is proportional to detection probability 
    for x in range(0, heatmap_pred.shape[-1], 2):
        for y in range(0, heatmap_pred.shape[-2], 2):
            arrow_color = int(heatmap_pred[y,x] * 255 * 10)
            out_image = cv2.arrowedLine(out_image, (x*scaleup, y*scaleup), 
                                        (int(x+flow_hm[0, y, x])*scaleup, 
                                         int(y+flow_hm[1, y, x])*scaleup), 
                                         (arrow_color,0,0), 1, 
                                         cv2.LINE_AA,0,0.3)
    
    # Draw gt, original position and arrow in green and destination in blue
    for (pres_pos, pos) in flow_pairs:
        out_image = cv2.arrowedLine(out_image, 
                                    (pres_pos[0]*scaleup, pres_pos[1]*scaleup), 
                                    (int(pos[0])*scaleup, int(pos[1])*scaleup), 
                                    (0,255,0), 2, 
                                    cv2.LINE_AA)
        out_image = cv2.drawMarker(out_image, 
                                   tuple(pres_pos*scaleup), 
                                   tuple([0,255,0]), 
                                   cv2.MARKER_SQUARE, 
                                   markerSize=10, 
                                   thickness=2)
        out_image = cv2.drawMarker(out_image, 
                                   tuple(pos*scaleup), tuple([0,0,255]), 
                                   cv2.MARKER_SQUARE, 
                                   markerSize=10, thickness=2)
    
    # Yellow diamond represents extracted detection from reconstruction
    for point in det_point_rec:
        out_image = cv2.drawMarker(out_image, tuple(point*scaleup), 
                                   tuple([255,255,0]), cv2.MARKER_DIAMOND, 
                                   markerSize=5, thickness=4)

    # Add a legend to the output image
    font_size = 0.8
    textSize, baseline = cv2.getTextSize("Pred points", 
                                         cv2.FONT_HERSHEY_SIMPLEX, 
                                         font_size, 1)
    out_image = cv2.rectangle(out_image, 
                              (out_image.shape[1] - textSize[0] - 30, 
                               int(out_image.shape[0] 
                                   - 2 * baseline - 8 * textSize[1] -15)), 
                              (out_image.shape[1], int(out_image.shape[0])), 
                              [175,175,175], -1)
    
    # Add a legend for the predicted offset
    cv2.arrowedLine(out_image, (out_image.shape[1] - textSize[0] - 25, 
                                int(out_image.shape[0] 
                                    - 17.5 * baseline - textSize[1] / 2)), 
                                (out_image.shape[1] - textSize[0] - 10, 
                                 int(out_image.shape[0] 
                                     - 17.5 * baseline - textSize[1] / 2)), 
                                     (255,0,0), 1, cv2.LINE_AA,0,0.3)
    
    cv2.putText(out_image, "Pred Offset", 
                (out_image.shape[1] - textSize[0] - 5, 
                 int(out_image.shape[0] - 17.5*baseline)), 
                 cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                 [255,255,255], thickness=2)
    
    # Add a legend for the reconstructed offset
    cv2.drawMarker(out_image, 
                   (out_image.shape[1] - textSize[0] - 15, 
                    int(out_image.shape[0] - 13.5*baseline - textSize[1] / 2)), 
                    tuple([255,255,0]), cv2.MARKER_DIAMOND, 
                    markerSize=5, thickness=4)
    
    cv2.putText(out_image, f"Rec {legend_rec}", 
                (out_image.shape[1] - textSize[0] - 5, 
                 int(out_image.shape[0] - 13.5*baseline)), 
                 cv2.FONT_HERSHEY_SIMPLEX, 
                 font_size, [255,255,255], thickness=2)

    # Add a legend for the ground truth offset
    cv2.arrowedLine(out_image, 
                    (out_image.shape[1] - textSize[0] - 25, 
                     int(out_image.shape[0] - 9.5*baseline - textSize[1] / 2)), 
                    (out_image.shape[1] - textSize[0] - 10, 
                     int(out_image.shape[0] - 9.5*baseline - textSize[1] / 2)), 
                     (0,255,0), 2, cv2.LINE_AA)
    
    cv2.putText(out_image, "Gt Offset", 
                (out_image.shape[1] - textSize[0] - 5, 
                 int(out_image.shape[0] - 9.5*baseline)), 
                 cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                 [255,255,255], thickness=2)
    
    # Add a legend for the ground truth detection
    cv2.drawMarker(out_image, 
                   (out_image.shape[1] - textSize[0] - 15, 
                    int(out_image.shape[0] - 5.5*baseline - textSize[1] / 2)), 
                    tuple([0,255,0]), cv2.MARKER_SQUARE, 
                    markerSize=10, thickness=2)
    
    cv2.putText(out_image, f"Gt {legend_det}", 
                (out_image.shape[1] - textSize[0] - 5, 
                 int(out_image.shape[0] - 5.5*baseline)), 
                 cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                 [255,255,255], thickness=2)
    
    # Add a legend for the reconstructed detection
    cv2.drawMarker(out_image, 
                   (out_image.shape[1] - textSize[0] - 15, 
                    int(out_image.shape[0] - 1.5 * baseline - textSize[1] / 2)), 
                    tuple([0,0,255]), 
                    cv2.MARKER_SQUARE, markerSize=10, thickness=2)
    
    cv2.putText(out_image, f"Gt {legend_rec}", 
                (out_image.shape[1] - textSize[0] - 5, 
                 int(out_image.shape[0] - 1.5*baseline)), 
                 cv2.FONT_HERSHEY_SIMPLEX, font_size, 
                 [255,255,255], thickness=2)
    
    # Add a title bar to the output image
    out_image = cv2.copyMakeBorder(out_image, 32, 0, 0, 0, 
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    textSize, baseline = cv2.getTextSize(title, 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    out_image = cv2.putText(out_image, title, 
                            (int(out_image.shape[1] / 2 - textSize[0] / 2), 23), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                            [255,255,255], thickness=1)

    # Return the output image
    return out_image

def visualize_density_cv2(input_img: np.ndarray, density_pred: np.ndarray, 
                          title: str, 
                          points: Optional[List[Tuple[int, int]]] = None, 
                          count: Optional[int] = None, 
                          count_det: Optional[int] = None, 
                          prec: Optional[float] = None, 
                          rec: Optional[float] = None, 
                          gt_points: Optional[List[Tuple[int, int]]] = None, 
                          roi: Optional[np.ndarray] = None, 
                          pred_scale: int = 4) -> np.ndarray:
    """
    Visualizes a density prediction on top of an input image.

    Args:
        input_img (np.ndarray): The input image to visualize.
        density_pred (np.ndarray): The density prediction to visualize.
        title (str): The title to display on the output image.
        points (Optional[List[Tuple[int, int]]]): The detected points to 
                                            display on the output image.
        count (Optional[int]): The total count of points in the input image.
        count_det (Optional[int]): The count of detected points in the 
                                            input image.
        prec (Optional[float]): The precision metric for the detected points.
        rec (Optional[float]): The recall metric for the detected points.
        gt_points (Optional[List[Tuple[int, int]]]): The ground truth points 
                                            to display on the output image.
        roi (Optional[np.ndarray]): The region of interest to display on 
                                            the output image.
        pred_scale (int): The scale factor to apply to the density prediction.

    Returns:
        np.ndarray: The output image with the density prediction visualized.
    """
    
    # Convert the density prediction and input image to uint8 format
    density_pred = np.array(density_pred * 255, dtype = np.uint8).squeeze()
    input_img = np.array(input_img * 255, dtype = np.uint8)
    
    # Normalize the density prediction to the range [0, 255]
    density_pred = cv2.normalize(density_pred, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX)

    # Resize the density prediction if necessary
    if pred_scale != 1:
        density_pred = cv2.resize(density_pred, 
                                  tuple([c*pred_scale 
                                         for c 
                                         in density_pred.shape[:2]][::-1]))

    # Apply a color map to the density prediction to create a heatmap image
    heatmap_img = cv2.applyColorMap(density_pred, 
                                    cv2.COLORMAP_RAINBOW)

    # Resize the input image if necessary
    if input_img.shape[:2] != heatmap_img.shape[:2]:
        input_img = cv2.resize(input_img, heatmap_img.shape[:2])

    # Combine the input image and heatmap image using a weighted sum
    out_image = cv2.addWeighted(input_img, 0.8, heatmap_img, 0.5, 0)
    
    # Add the region of interest to the output image if available
    if roi is not None:
        roi = np.array(roi.squeeze() * 255, dtype = np.uint8)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE )
        if pred_scale != 1:
            roi = cv2.resize(roi, tuple([c*pred_scale 
                                         for c 
                                         in roi.shape[:2]][::-1]), 
                                         interpolation=cv2.INTER_NEAREST)
        out_image = cv2.addWeighted(out_image, 0.8, roi, 0.4, 0)
        
    # Draw detected points on the output image if available
    if points is not None:
        for point in points:
            out_image = cv2.drawMarker(out_image, 
                                       tuple(point*pred_scale), 
                                       tuple([190,0,0]), 
                                       cv2.MARKER_CROSS, 
                                       markerSize=4, 
                                       thickness=1)
    
        textSize, baseline = cv2.getTextSize("Pred points", 
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        
        cv2.drawMarker(out_image, 
                       (out_image.shape[1] - textSize[0] - 15, 
                        int(out_image.shape[0] - 2*baseline - 2 * textSize[1])), 
                        [190,0,0], cv2.MARKER_CROSS, markerSize=4, thickness=1)
        
        cv2.putText(out_image, "Pred points", 
                    (out_image.shape[1] - textSize[0] - 5, 
                     int(out_image.shape[0] - 2*baseline - 1.5*textSize[1])), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255,255,255], thickness=1)

        out_image = cv2.rectangle(out_image, 
                                  (out_image.shape[1] - textSize[0] - 30, 
                                   int(out_image.shape[0] 
                                       - 2 * baseline - 2 * textSize[1] -15)), 
                                  (out_image.shape[1], int(out_image.shape[0])), 
                                  [255,255,255], 1)
    
    # Draw ground truth points on the output image if available
    if gt_points is not None:
        for point in gt_points:
            out_image = cv2.drawMarker(out_image, tuple(point*pred_scale), 
                                       [0,190,0], cv2.MARKER_SQUARE, 
                                       markerSize=4, thickness=1)
    
        cv2.drawMarker(out_image, 
                       (out_image.shape[1] - textSize[0] - 15, 
                        int(out_image.shape[0] 
                            - 1.5 * baseline - textSize[1] / 2)), 
                            [0,190,0], cv2.MARKER_SQUARE, markerSize=4, 
                            thickness=1)
        cv2.putText(out_image, "Gt points", 
                    (out_image.shape[1] - textSize[0] - 5, 
                     int(out_image.shape[0] - 1.5*baseline)), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                     [255,255,255], thickness=1)
    
    # Add a margin and captions to the output image
    out_image = cv2.copyMakeBorder(out_image, 32, 32, 0, 0, 
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # Add a title bar to the output image
    textSize, baseline = cv2.getTextSize(title, 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    out_image = cv2.putText(out_image, title, 
                            (int(out_image.shape[1] / 2 - textSize[0] / 2), 23), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,255,255], 
                            thickness=1)
    
    # Add bottom text to the output image if available
    if count is not None and count_det is not None:
        count = round(min(count, 999))
        count_det = round(min(count_det, 999))

        bottom = f"Count - Hm {count} - Det {count_det} "
    else:
        bottom = ""
        
    if gt_points is not None:
        bottom = bottom + f"Gt - {str(len(gt_points))} "

    if prec is not None:
        bottom = bottom + f"|| Prec - {prec:.2f} Rec - {rec:.2f}"
    
    textSize, baseline = cv2.getTextSize(bottom, 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    out_image = cv2.putText(out_image, bottom, 
                            (int(out_image.shape[1] / 2 - textSize[0] / 2), 
                             out_image.shape[2]-15), 
                             cv2.FONT_HERSHEY_SIMPLEX, 
                             0.5, [255,255,255], thickness=1)
    
    # Return the output image
    return out_image

# adapted from 
# https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py

def generate_colors(N: int) -> List[Tuple[int, int, int]]:
    """
    Generate random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Parameters:
    - N: An integer representing the number of colors to generate.

    Returns:
    - A list of tuples representing the RGB values for each color.
    """
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple(round(i * 255) 
                                      for i in colorsys.hsv_to_rgb(*c)), hsv))
    perm = np.random.permutation(N)
    colors = [colors[idx] for idx in perm]
    
    return colors


def make_tracking_vis(frames: List[np.ndarray], 
                      full_tracks: List[List[Tuple[int, Tuple[int, int]]]], 
                      title: str, roi: List[np.ndarray], pred_scale: int
                      ) -> List[np.ndarray]:
    """
    Create a visualization of object tracks on a sequence of frames.

    Parameters:
    - frames: A list of NumPy arrays representing the input frames.
    - full_tracks: A list of lists, where each inner list contains tuples 
                        representing the full track for an object.
    - title: A string representing the title of the visualization.
    - roi: A list of NumPy arrays representing regions of interest 
                        for each frame (default is None).
    - pred_scale: An integer representing the scale factor for the 
                        partial tracks (default is 4).

    Returns:
    - A list of NumPy arrays representing the visualizations for each frame.
    """
    out_frames = list()
    
    colors = generate_colors(len(full_tracks))

    for i, frame in enumerate(frames):
        
        # frame_inv = inverse_img_norm(frame)
        partial_tracks = []
        for track_id, full_track in enumerate(full_tracks):
            partial_tracks.append((track_id, get_local_track(full_track, i)))
        
        if roi is not None:
            roi_curr = roi[i]
        else:
            roi_curr = None

        vis = make_frame_visu_track(frame, partial_tracks, colors, 
                                    title=title, roi=roi_curr, 
                                    pred_scale=pred_scale)
        
        out_frames.append(vis)
        
    return out_frames


def get_local_track(full_track: List[Tuple[int, Tuple[int, int]]], 
                    timestamp: int, local_size: int = 10
                    ) -> List[Tuple[int, int]]:
    """
    Get a local track for an object at a given timestamp.

    Parameters:
    - full_track: A list of tuples representing the full track for an object.
    - timestamp: An integer representing the timestamp for the local track.
    - local_size: An integer representing the size of the local track 
                    (default is 10).

    Returns:
    - A list of tuples representing the local track for the object.
    """
    
    if timestamp < full_track[0][0] or timestamp > full_track[-1][0]:
        return []
    else:
        track_id = timestamp - full_track[0][0]
        return [track_point[1] 
                for track_point 
                in full_track[max(0,track_id-local_size):track_id]]           


def make_frame_visu_track(input_img: np.ndarray, 
                          partial_tracks: List[Tuple[int, np.ndarray]], 
                          colors: List[Tuple[int, int, int]], 
                          title: str, roi: np.ndarray = None, 
                          pred_scale: int = 4) -> np.ndarray:
    """
    Create a visualization of object tracks on an image.

    Parameters:
    - input_img: A NumPy array representing the input image.
    - partial_tracks: A list of tuples, where each tuple contains an object 
                        ID and a NumPy array representing a partial track 
                        for that object.
    - colors: A list of tuples, where each tuple contains the RGB color values 
                        for an object.
    - title: A string representing the title of the visualization.
    - roi: An optional NumPy array representing a region of interest 
                        (default is None).
    - pred_scale: An integer representing the scale factor for the 
                        partial tracks (default is 4).

    Returns:
    - A NumPy array representing the visualization.
    """
    
    out_image = np.ascontiguousarray(np.array(input_img * 255, 
                                              dtype = np.uint8))
    
    
    if roi is not None:
        roi = np.array(roi.squeeze() * 255, dtype = np.uint8)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE )
        if pred_scale != 1:
            roi = cv2.resize(roi, 
                             tuple([c*pred_scale for c in roi.shape[:2][::-1]]), 
                             interpolation=cv2.INTER_NEAREST)
        out_image = cv2.addWeighted(out_image, 0.8, roi, 0.2, 0)
    
    for track_id, partial_track in partial_tracks:
        partial_track = np.array(partial_track, np.int32)*pred_scale
        
        if partial_track.shape[0] != 0:
            cv2.drawMarker(out_image, 
                           (partial_track[-1,0], partial_track[-1,1]), 
                           colors[track_id], cv2.MARKER_STAR, 
                           markerSize=6, thickness=2) 
                        # MarkerType[, markerSize[, thickness[, line_type]]]])

            cv2.putText(out_image, 
                        str(track_id), 
                        (partial_track[-1,0] + 8, partial_track[-1,1]+3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, 
                        colors[track_id], thickness=1)
            
            cv2.polylines(out_image, [partial_track.reshape((-1, 1, 2))], 
                          False, colors[track_id], thickness=2)

            # ax.plot(partial_track[:,0], partial_track[:,1], 
            #         linewidth=5, c=colormap((track_id*10) % nb_track))
            

        
    out_image = cv2.copyMakeBorder(out_image, 32, 32, 0, 0, 
                                   cv2.BORDER_CONSTANT, value=[0,0,0])
    
    #Title Bar
    textSize, baseline = cv2.getTextSize(title, 
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    out_image = cv2.putText(out_image, title, 
                            (int(out_image.shape[1] / 2 - textSize[0] / 2), 23), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,255,255], 
                            thickness=1)
    
    # # Bottom text add gt and metric if available
    # if count is not None and count_det is not None:
    #     bottom = f"Count - Hm {count} - Det {count_det} "
    # else:
    #     bottom = ""
        
    # if gt_points is not None:
    #     bottom = bottom + f"Gt - {str(gt_points.shape[0])} "

    # if prec is not None:
    #     bottom = bottom + f"|| Prec - {prec} Rec - {rec}"
    
    # textSize, baseline = cv2.getTextSize(bottom, 
    #                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # out_image = cv2.putText(out_image, bottom, 
    #                     (int(out_image.shape[1] / 2 - textSize[0] / 2), 
    #                         561), 
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
    #                         [255,255,255], thickness=1)
    
    return out_image
    

def inverse_img_norm(img: torch.Tensor, 
                     mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
                     ) -> np.ndarray:
    """
    Convert a normalized PyTorch tensor back to a numpy image.

    Parameters:
    - img: A PyTorch tensor representing the normalized image.
    - mean: A tuple of three floats representing the mean values used 
                for normalization (default is (0.485, 0.456, 0.406)).
    - std: A tuple of three floats representing the standard deviation values 
                used for normalization (default is (0.229, 0.224, 0.225)).

    Returns:
    - A numpy array representing the unnormalized image.
    """
    #Convert normalize pytorch tensor back to numpy image

    inv_normalize = transforms.Normalize(
        mean=[-mean[0]/0.229, -mean[1]/0.224, -mean[2]/0.225],
        std=[1/std[0], 1/std[1], 1/std[2]]
    )

    img_unorm = torch.clip(inv_normalize(img), 0, 1).squeeze().permute(1,2,0)

    return img_unorm.cpu().numpy()


def save_visualization_as_video(project_root: str, 
                                dict_visualization: Dict[str, List[np.ndarray]], 
                                model_id: str, epoch: int, 
                                out_type: str = "avi") -> None:
    """
    Save a visualization as a video file.

    Parameters:
    - project_root: A string representing the root directory of the project.
    - dict_visualization: A dictionary mapping visualization types to 
                            lists of frames.
    - model_id: A string representing the ID of the model.
    - epoch: An integer representing the epoch number.
    - out_type: A string representing the output file type (default is "avi").

    Returns:
    - None
    """
    
    for visu_type, frame_list in dict_visualization.items():
        file_name = (project_root 
            + f"/results/{model_id}/{model_id}_epoch_{str(epoch)}_{visu_type}")

        if out_type == "avi":
            save_video_avi(frame_list, file_name)
        elif out_type == "mp4":
            save_video_mp4(frame_list, file_name)

def save_video_mp4(frame_list: List[np.ndarray], path: str, 
                   save_framerate: int = 30) -> None:
    """
    Save a list of frames as an MP4 video file.

    Parameters:
    - frame_list: A list of numpy arrays representing the frames.
    - path: A string representing the output file path.
    - save_framerate: An integer representing the frame rate of the 
                        output video (default is 30).

    Returns:
    - None
    """
    file_path = Path('{}.mp4'.format(path))
    file_path.parents[0].mkdir(parents=True, exist_ok=True)

    imageio.mimwrite(file_path, frame_list, fps=save_framerate, 
                     macro_block_size=1)        

def save_video_avi(frames: List[np.ndarray], path: str, 
                   save_framerate: int = 30) -> None:
    """
    Save a list of frames as an AVI video file.

    Parameters:
    - frames: A list of numpy arrays representing the frames.
    - path: A string representing the output file path.
    - save_framerate: An integer representing the frame rate of the 
                        output video (default is 30).

    Returns:
    - None
    """
    video_h, video_w = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    file_path = Path('{}.avi'.format(path))
    
    #Check if parent dir exist otherwise make it
    file_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    out = cv2.VideoWriter(str(file_path), fourcc, save_framerate, 
                          (video_w, video_h))
    
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()


def visualize_density(input_img, density_pred, roi=None):
      
    my_dpi=50
    fig = plt.figure(figsize=(float(input_img.shape[1])
                              / my_dpi,
                              float(input_img.shape[0])
                              / my_dpi))
    canvas = FigureCanvasAgg(fig)
    
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    
    ax.imshow(input_img)
    ax.imshow(density_pred, alpha=0.4, cmap='rainbow')
    if roi is not None:
        ax.imshow(roi, alpha=0.2)
    
    canvas.draw()
    data = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    
    return cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
    

def visualize_count(input_img: np.ndarray, density_pred: np.ndarray, 
                    grid_size: int = 40, 
                    roi: Optional[Tuple[int, int, int, int]] = None
                    ) -> Tuple[np.ndarray, int]:
    """
    Visualize the crowd count on an image.

    Parameters:
    - input_img: A NumPy array representing the input image.
    - density_pred: A NumPy array representing the density map.
    - grid_size: The size of the grid cells in pixels (default is 40).
    - roi: An optional tuple representing the region of interest (x, y, w, h).

    Returns:
    - A tuple containing the output image and the crowd count.
    """

    my_dpi=50
    
    # Set up figure
    fig=plt.figure(figsize=(float(input_img.shape[1])
                            / my_dpi,
                            float(input_img.shape[0])
                            / my_dpi))
    canvas = FigureCanvasAgg(fig)
    
    ax=fig.add_subplot(111)

    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

    # Set the gridding interval: here we use the major tick interval
    locx = plticker.MultipleLocator(base=grid_size)
    # plticker.LinearLocator(numticks=input_img.shape[1]//grid_size)
    locy = plticker.MultipleLocator(base=grid_size)
    # plticker.LinearLocator(numticks=input_img.shape[0]//grid_size)
    
    ax.xaxis.set_major_locator(locx)
    ax.yaxis.set_major_locator(locy)

    # Add the grid
    ax.grid(which='major', axis='both', linestyle='-', linewidth=1, 
            color="black")
    
    # Add the image
    ax.imshow(input_img)
    
    if roi is not None:
        ax.imshow(roi, alpha=0.2)

    # Find number of gridsquares in x and y direction
    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(grid_size)))
    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(grid_size)))

    # Add some labels to the gridsquares
    for j in range(ny):
        y=grid_size/2+j*grid_size
        for i in range(nx):
            x=grid_size/2.+float(i)*grid_size
            count = np.abs(np.sum(density_pred[j * grid_size : 
                                               j * grid_size + grid_size, 
                                               i * grid_size :
                                               i * grid_size + grid_size]))
            if count > 0.1:
                ax.text(x,y,'{:.1f}'.format(count).lstrip('0'),
                        color='red', ha='center', va='center', 
                        fontsize=25, alpha=0.8)
    
    canvas.draw()
    data = np.array(canvas.renderer.buffer_rgba())
    plt.close()
    
    return cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)


def visualize_gt_head_feet(img: np.ndarray, gt_view: list, 
                           display: bool = False) -> np.ndarray:
    """
    Visualize the ground truth head and feet markers on an image.

    Parameters:
    - img: A NumPy array representing an image.
    - gt_view: A list of GroundTruthView objects.
    - display: A boolean indicating whether to display the image.

    Returns:
    - A NumPy array representing the image with ground truth head and 
                    feet markers.
    """
    
    img = img.copy()
    
    for gt in gt_view:
        color = (gt.id * 67 % 255, (gt.id + 1) * 36 % 255 , 167)
        
        cv2.drawMarker(img, tuple(gt.feet), color, 
                       markerType=cv2.MARKER_CROSS, 
                       markerSize=20, thickness=2)
        cv2.drawMarker(img, tuple(gt.head), color, 
                       markerType=cv2.MARKER_STAR, 
                       markerSize=20, thickness=2)

    if display:    
        plt.imshow(img)
        plt.show()

    return img