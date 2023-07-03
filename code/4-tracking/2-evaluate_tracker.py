"""
Look into adding nb_gt into the compute mot metric function
look at metric.py for nb_gt calculation.

"""


import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import cv2
import numpy as np
import pandas as pd
import configparser

from pathlib import Path

from collections import defaultdict
from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string

from utils.multiview_utils import MultiviewVids
from utils.storage_utils import TrackingHDF5
from utils.coordinate_utils import project_to_ground_plane_cv2

from detection.dataset.utils import generate_scene_roi_from_view_rois
from detection.dataset.pomelo import PomeloSup
from detection.configs.arguments import get_train_config_dict
from detection.inference import start_inference
from detection.misc.metric import  compute_mot_metric
from detection.misc.utils import  flatten

from tracker.visualization import make_tracking_vis
from utils.visualization import save_video_mp4

def convert_ann_to_tracks(gt_anns_list):
    track_as_list = []

    for gt_ann in gt_anns_list:
         track_as_list.append({'FrameId':gt_ann.frame_id, 'Id':int(gt_ann.person_id), 'X':int(gt_ann.feet[0]), 'Y':int(gt_ann.feet[1])})

    return track_as_list

def make_dataframe_from_tracks_list(track_list):
    
    tracks_as_df = pd.DataFrame(track_list)
    # print(tracks_as_df)
    if  tracks_as_df.empty:
        tracks_as_df = pd.DataFrame(columns =['FrameId','Id','X','Y'])
    
    tracks_as_df = tracks_as_df.set_index(['FrameId', 'Id'])
    
    return tracks_as_df

if __name__ == '__main__':
    log.info("2 - Starting evaluation of tracking results on validation set-")

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

    dset = PomeloSup(train_conf["data_conf"], "val")
    gt_ground = flatten([dset.get_gt(index) for index in range(len(dset))]) #mvv[0].extract(range(end_ind))

    if len(gt_ground) == 0:
        log.error("No ground truth found. Please annotate the validation set and run tracker in evaluation mode.")
        sys.exit(0)

    start_ind_track = 0
    end_ind_track = len(dset)

    if train_conf["data_conf"]["mode"] == "evaluation":
        storage_track_path = track_dir / f'{train_conf["main"]["name"]}_frames_{start_ind_track}_{end_ind_track}_tracking_evaluation.hdf5'
    else:
         log.error("Tracker evaluation not possible for inference mode.")

    storages_track = TrackingHDF5(storage_track_path)

    frame_original_size = tuple(mvv.get_frame_size())

    frame_input_size = train_conf["data_conf"]["frame_input_size"]
    homography_input_size = train_conf["data_conf"]["homography_input_size"]
    homography_output_size = train_conf["data_conf"]["homography_output_size"]
    hm_size = train_conf["data_conf"]["hm_size"]

    homographies = mvv.get_homographies(homography_input_size, homography_output_size)       

    pred_tracks = storages_track.get_tracks_as_df()
    gt_tracks = convert_ann_to_tracks(gt_ground)

    pred_tracks = make_dataframe_from_tracks_list(pred_tracks)
    gt_tracks = make_dataframe_from_tracks_list(gt_tracks)


    tracking_metric =  compute_mot_metric(gt_tracks, pred_tracks, conf_dict["tracking"]["metric_threshold"])

    log.info(dict_to_string(tracking_metric))

    log.info("1 - Visualization completed -")