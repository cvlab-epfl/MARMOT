import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import cv2
import numpy as np
import configparser

from pathlib import Path

from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string

from utils.multiview_utils import MultiviewVids
from utils.storage_utils import InferenceHDF5
from utils.coordinate_utils import project_to_ground_plane_cv2

from detection.dataset.utils import generate_scene_roi_from_view_rois
from detection.configs.arguments import get_train_config_dict
from detection.inference import start_inference

from utils.visualization import visualize_gt_cv2, save_video_mp4

if __name__ == '__main__':
    log.info("1 - Starting visualization off inference results -")

    conf_dict =  get_config_dict()
    log.debug(f'conf_dict: {dict_to_string(conf_dict["inference"])}')
    root_dir = Path(conf_dict["main"]["data_root"]) / "3-inference/"
    
    #detection model configuration
    train_conf = get_train_config_dict(conf_dict["training"]["train_config_file"])  

    mvv = MultiviewVids() 
    
    storage_path = root_dir / f'{train_conf["main"]["name"]}_inference.hdf5'

    storage = InferenceHDF5(storage_path, None, None)

    frame_original_size = tuple(mvv.get_frame_size())

    frame_input_size = frame_original_size#train_conf["data_conf"]["frame_input_size"]
    homography_input_size = frame_original_size#train_conf["data_conf"]["homography_input_size"]
    homography_output_size = train_conf["data_conf"]["homography_output_size"]
    hm_size = train_conf["data_conf"]["hm_size"]

    homographies = mvv.get_homographies(homography_input_size, homography_output_size)       
    ROIs = mvv.get_view_ROIs()
    
    ROI_mask, ROI_boundary = generate_scene_roi_from_view_rois(ROIs, homographies, frame_original_size, homography_input_size, homography_output_size, hm_size)

    start_ind, end_ind = conf_dict["inference"]["inference_visualize"]

    log.info(f"Found a total of {storage.get_nb_frames()} frames in the inference storage. Will visualize frames from {start_ind} to {end_ind}")
    log.info(f"Camera sequence contains at most {mvv.get_max_frame_id()} frames for camera 0 {mvv[0].get_max_frame_id()}")
    if end_ind == -1:
        end_ind = storage.get_nb_frames()

    video_out = []
    frames = mvv[0].extract(list(range(start_ind, end_ind)), rgb=True)
    for index in range(start_ind, end_ind):
        log.info(f"Visualizing frame {index}")
        frame = frames[index-start_ind]
         #mvv[0].extract([index])[0]
        gp_det, gp_score, _ = storage.get_detection(index, get_reid=False)
        frame_ground = project_to_ground_plane_cv2(frame, homographies[0], tuple(homography_output_size))

        #filter ou gp_det according to gp_score, remove if value smaller than threshold 
        gp_det = gp_det[gp_score > conf_dict["inference"]["detection_threshold"]]
        #count detection removed
        if len(gp_score) - len(gp_det) > 0:
            log.debug(f"Removed {len(gp_score) - len(gp_det)} detections with score smaller than {conf_dict['inference']['detection_threshold']}")


        frame_with_pred = visualize_gt_cv2(frame_ground, gt_points=None, roi=ROI_mask, pred_points=gp_det)
        video_out.append(frame_with_pred)

    save_dir = root_dir / "visualisation"

    save_video_mp4(video_out, save_dir / f"inference_pred_{start_ind}_{end_ind}", save_framerate=conf_dict["main"]["save_framerate"])

        #saving heatmap as png
    for det_type in train_conf["training"]["detection_to_evaluate"]:
        heatmap = storage.get_heatmaps(det_type, start_ind, end_ind, agg="avg")
        #normalize heatmap between 0 and 1
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        # heatmap = np.transpose(heatmap, (1, 2, 0))
        # heatmap = cv2.resize(heatmap, frame_original_size)
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        cv2.imwrite(str(save_dir / f"evaluation_visualization_{det_type}_{start_ind}_{end_ind}.png"), heatmap)

    log.info("1 - Visualization completed -")
    