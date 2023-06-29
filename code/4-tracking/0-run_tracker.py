import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import cv2
import numpy as np
import configparser

from multiprocessing import Queue
from pathlib import Path

from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string


from utils.multiview_utils import MultiviewVids
from utils.storage_utils import InferenceHDF5, TrackingHDF5
from utils.coordinate_utils import project_to_ground_plane_cv2

from detection.dataset.utils import generate_scene_roi_from_view_rois
from detection.configs.arguments import get_train_config_dict
from detection.inference import start_inference

from tracker import tracker

from utils.visualization import visualize_gt_cv2, save_video_mp4

if __name__ == '__main__':
    log.info("0 - Starting to run tracker -")

    conf_dict =  get_config_dict()
    log.debug(f'conf_dict: {dict_to_string(conf_dict["tracking"])}')
    root_dir = Path(conf_dict["main"]["data_root"])
    inf_dir = root_dir / "3-inference/"
    track_dir = root_dir / "4-tracking/"

    use_appearance = conf_dict["tracking"]["reid"]
    
    #detection model configuration
    train_conf = get_train_config_dict(conf_dict["training"]["train_config_file"])  


    if conf_dict["tracking"]["mode"] == "evaluation":
        storage_path = root_dir / "2-training" / f'{train_conf["main"]["name"]}_evaluation.hdf5'
    elif conf_dict["tracking"]["mode"] == "inference":
        storage_path = inf_dir / f'{train_conf["main"]["name"]}_inference.hdf5'
    
    storage = InferenceHDF5(storage_path, None, None)

    start_ind, end_ind = conf_dict["tracking"]["tracker_range"]
    if end_ind == -1:
        end_ind = storage.get_nb_frames()
             
    tracker_step = conf_dict["tracking"]["tracker_step"]

    process_queues = Queue()
    result_queues = Queue()
            
            #initialize tracker
    mussp_tracker = tracker.MuSSPTracker(train_conf, process_queues, use_appearance, result_queue=result_queues)
    mussp_tracker.start()

    log.debug(f"Starting to run tracker on frame {start_ind} to {end_ind}")

    video_out = []
    for index in range(start_ind, end_ind, tracker_step):
        preds_in_window = []
        for i in range(index, min(index+tracker_step, end_ind)):
            gp_det, gp_score, gp_app = storage.get_detection(i, get_reid=True)

            #filter ou gp_det according to gp_score, remove if value smaller than threshold 
            gp_det = gp_det[gp_score > conf_dict["inference"]["detection_threshold"]]
            gp_app = gp_app[gp_score > conf_dict["inference"]["detection_threshold"]]

            if len(gp_score) - len(gp_det) > 0:
                log.debug(f"Removed {len(gp_score) - len(gp_det)} detections with score smaller than {conf_dict['inference']['detection_threshold']}")
            
            preds_in_window.append((gp_det, gp_app))

        log.debug("Sending to tracker")
        process_queues.put(preds_in_window)

    
    process_queues.put(-1)
    log.debug("Waiting for tracker to finish")
    #wait for all processes to finish
    tracks = result_queues.get(block=True)

    #save tracks to storage
    if conf_dict["tracking"]["mode"] == "evaluation":
        storage_track_path = track_dir / f'{train_conf["main"]["name"]}_frames_{start_ind}_{end_ind}_tracking_evaluation.hdf5'
    elif conf_dict["tracking"]["mode"] == "inference":
        storage_track_path = track_dir / f'{train_conf["main"]["name"]}_frames_{start_ind}_{end_ind}_tracking_inference.hdf5'

    storages_track = TrackingHDF5(storage_track_path)

    storages_track.write_tracks(tracks)

       

    log.info(f"0 - Tracking completed for frame {start_ind} to {end_ind}")
    