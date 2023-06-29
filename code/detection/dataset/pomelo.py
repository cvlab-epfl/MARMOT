import os
import json

import numpy as np

from pathlib import Path

from detection.dataset.utils import get_frame_from_file
from detection.dataset.sceneset import SceneBaseSet

from utils.log_utils import log

class PomeloSup(SceneBaseSet):
    def __init__(self, data_conf, set_type):
        super().__init__(data_conf)
                
        root_dir = Path(self.pipeline_conf["main"]["data_root"]) / "1-annotation/"
        self.nb_cam = self.mvv.get_nb_cams()
        self.cams = self.mvv.get_cameras()
        base_cam = self.cams[0]

        self.frame_dir_path = root_dir / set_type
        self.frame_list_path = [int(path.stem.split("_")[1]) for path in (self.frame_dir_path / base_cam.name).glob('*.png')]
        self.frame_list_path.sort()

        self.multiview_gt = self.mvv.load_gt(set_type)

        self.nb_frames = len(self.frame_list_path)
    
        log.debug(f"Dataset Pomelo set {set_type} containing {self.nb_frames} frames from {self.get_nb_view()} views ")

        #use parent class to generate ROI and occluded area maps
        self.generate_scene_elements()
        self.log_init_completed()

    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """

            frame_path = self.frame_dir_path / self.cams[view_id].name / f"{self.cams[view_id].name}_{self.frame_list_path[index]}.png"

            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = get_frame_from_file(frame_path)
            frame = self.cams[view_id].undistort([frame])[0]

            return frame

    def _get_gt(self, index, view_id):
        return self.multiview_gt[view_id][index]    
    
    def get_nb_view(self):
        return self.nb_cam
    
    def get_ann_length(self, view_id):
        return len(self.multiview_gt[view_id]) if len(self.multiview_gt) > 0 else 0
        
    def get_length(self):
        return self.nb_frames


class PomeloInf(SceneBaseSet):
    def __init__(self, data_conf):
        super().__init__(data_conf)
                
        root_dir = Path(self.pipeline_conf["main"]["data_root"]) / "1-annotation/"
        self.nb_cam = self.mvv.get_nb_cams()
        self.cams = self.mvv.get_cameras()

        self.nb_frames = self.mvv.get_max_frame_id()
    
        log.debug(f"Dataset Pomelo inference containing {self.nb_frames} frames from {self.get_nb_view()} views ")

        #use parent class to generate ROI and occluded area maps
        self.generate_scene_elements()
        self.log_init_completed()

    def get_frame(self, index, view_id):
            """
            Read and return undistoreted frame coresponding to index and view_id.
            The frame is return at the original resolution
            """

            # log.debug(f"pomelo dataset get frame {index} {view_id}")
            frame = self.cams[view_id].extract([index], rgb=True)[0]
            frame = self.cams[view_id].undistort([frame])[0]

            return frame

    def _get_gt(self, index, view_id):
        return []
    
    def get_nb_view(self):
        return self.nb_cam
        
    def get_length(self):
        return self.nb_frames