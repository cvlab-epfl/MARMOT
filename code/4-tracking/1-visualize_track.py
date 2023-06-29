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
from utils.storage_utils import TrackingHDF5
from utils.coordinate_utils import project_to_ground_plane_cv2

from detection.dataset.utils import generate_scene_roi_from_view_rois
from detection.configs.arguments import get_train_config_dict
from detection.inference import start_inference

from tracker.visualization import make_tracking_vis
from utils.visualization import save_video_mp4

if __name__ == '__main__':
    log.info("1 - Starting visualization of tracking results -")

    conf_dict =  get_config_dict()
    log.debug(f'conf_dict: {dict_to_string(conf_dict["tracking"])}')
    root_dir = Path(conf_dict["main"]["data_root"]) / "2-training/"
    track_dir = Path(conf_dict["main"]["data_root"]) / "4-tracking/"
    
    #detection model configuration
    train_conf = get_train_config_dict(conf_dict["training"]["train_config_file"])  

    mvv = MultiviewVids() 
    
    train_conf["data_conf"]["pipeline_conf"] = conf_dict
    train_conf["data_conf"]["pipeline_mvv"] = mvv
    train_conf["data_conf"]["mode"] = "evaluation"
    
    if conf_dict["tracking"]["mode"] == "evaluation":
        from detection.dataset.pomelo import PomeloSup

        dset = PomeloSup(train_conf["data_conf"], "val")
        frames = [dset.get_frame(index, 0) for index in range(len(dset))] #mvv[0].extract(range(end_ind))

        start_ind_track = 0
        end_ind_track = len(frames)
    else:
        start_ind_track, end_ind_track = conf_dict["tracking"]["tracker_range"]
        if end_ind_track == -1:
            end_ind_track = mvv.get_max_frame_id()

    if conf_dict["tracking"]["mode"] == "evaluation":
        storage_track_path = track_dir / f'{train_conf["main"]["name"]}_frames_{start_ind_track}_{end_ind_track}_tracking_evaluation.hdf5'
    elif conf_dict["tracking"]["mode"] == "inference":
        storage_track_path = track_dir / f'{train_conf["main"]["name"]}_frames_{start_ind_track}_{end_ind_track}_tracking_inference.hdf5'
    
    storages_track = TrackingHDF5(storage_track_path)

    frame_original_size = tuple(mvv.get_frame_size())

    frame_input_size = frame_original_size# train_conf["data_conf"]["frame_input_size"]
    homography_input_size = frame_original_size#train_conf["data_conf"]["homography_input_size"]
    homography_output_size = train_conf["data_conf"]["homography_output_size"]
    hm_size = train_conf["data_conf"]["hm_size"]

    homographies = mvv.get_homographies(homography_input_size, homography_output_size)       
    ROIs = mvv.get_view_ROIs()
    
    ROI_mask, ROI_boundary = generate_scene_roi_from_view_rois(ROIs, homographies, frame_original_size, homography_input_size, homography_output_size, hm_size)

    start_ind, end_ind = conf_dict["tracking"]["tracker_visualize"]
    if end_ind == -1:
        end_ind = end_ind_track

    assert start_ind >= start_ind_track and end_ind <= end_ind_track, "Visualize range must be within tracking range"

    tracks, frame_id = storages_track.get_tracks(start_ind, end_ind)

    if conf_dict["tracking"]["mode"] == "inference":
        frames = mvv[0].extract(range(start_ind, end_ind), rgb=True)

    frames_ground = [project_to_ground_plane_cv2(frame, homographies[0], tuple(homography_output_size)) for frame in frames]

    video_out = make_tracking_vis(frames_ground, start_ind, tracks, frame_id, start_ind_track, ROI_mask)

    if conf_dict["tracking"]["mode"] == "evaluation":
        output_path = track_dir / "visualisation" / f"tracking_evaluation_pred_{start_ind}_{end_ind}"
    elif conf_dict["tracking"]["mode"] == "inference":
        output_path = track_dir / "visualisation" / f"tracking_inference_pred_{start_ind}_{end_ind}"
    
    save_video_mp4(video_out, output_path, save_framerate=conf_dict["main"]["save_framerate"])

    log.info("1 - Visualization completed -")
    