import uuid
import numpy as np
from detection.dataset.utils import aggregate_multi_view_gt, extract_points_from_gt
from utils.log_utils import log

class SceneBaseSet():
    """
    Parent class containing all the information regarding a single scene:
        - frames
        - groundtruth
        - Homography to groundplane
        - Region of interest
        - Occlusion mask

    The scene can contains multiple view. This class provide generic implementation and helper function.
    Specific function can be overwritten by child class: for example homography can be computed from camera calibration or point in the image depending on the type of scene.
    """

    def __init__(self, data_conf):
        super(SceneBaseSet, self).__init__()
        self.data_conf = data_conf

        self.mvv = data_conf["pipeline_mvv"]
        self.pipeline_conf = data_conf["pipeline_conf"]
        
        self.frame_original_size = tuple(self.mvv.get_frame_size())

        self.frame_input_size = data_conf["frame_input_size"]
        self.homography_input_size = data_conf["homography_input_size"]
        self.homography_output_size = data_conf["homography_output_size"]
        self.hm_size = data_conf["hm_size"]
        self.hm_image_size = data_conf["hm_image_size"]

        self.view_IDs = data_conf["view_ids"]

        self.scene_id = uuid.uuid4()

    def generate_scene_elements(self):
        self.homographies = self.mvv.get_homographies(self.homography_input_size, self.homography_output_size)       
        self.ROIs = self.mvv.get_view_ROIs()

    def log_init_completed(self):
        log.debug(f"Dataset from directory {self.pipeline_conf['main']['data_root']} containing {len(self)} frames loaded")
        log.debug(f"Dataset contains {self.get_nb_view()} view, the following are used: {self.view_IDs}")
        
    def get(self, index, view_id):

        frame = self.get_frame(index, view_id)
        homography = self.get_homography(view_id)

        return frame, homography

    def get_frame(self, index, view_id):
        return []

    def get_gt(self, index):

        gts = [self._get_gt(index, view_id) for view_id in self.view_IDs] 
        aggregated_gt = aggregate_multi_view_gt(gts, self.homographies, self.frame_original_size, self.homography_input_size, self.homography_output_size, self.hm_size)
        
        return aggregated_gt

    def get_gt_image(self, index, view_id):
        index = index

        gt = self._get_gt(index, view_id)
        gt_points, person_id = extract_points_from_gt(gt, self.hm_image_size, gt_original_size=self.frame_original_size)

        return gt_points, person_id

    def get_homography(self, view_id):
        return self.homographies[view_id]

    def get_ROI(self, view_id):
        return self.ROIs[view_id]

    def get_nb_used_view(self):
        return len(self.view_IDs)

    def get_length(self):
        log.error("Abstract class get_lentgth as be called, it should be overwriten it in child class")
        return -1

    def __len__(self):
        return self.get_length()