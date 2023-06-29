import h5py
import numpy as np
from pathlib import Path

from utils.log_utils import log

def flatten(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

class InferenceHDF5:
    """
    A class to store inference results in a HDF5 file.
    """
    def __init__(self, path, detection_size, reid_size):
        super(InferenceHDF5, self).__init__()
        
        if type(path) == str:
            path = Path(path)
            
        self.file_path = path
        self.detection_size = detection_size
        self.reid_size = reid_size
        
        self.create_file()
            
    def create_file(self):
        if not(self.file_path.is_file()):
            log.debug(f"Creating HDF5 file {self.file_path}")
            self.file_path.parents[0].mkdir(parents=True, exist_ok=True)
            with h5py.File(self.file_path, 'w') as f:
                evaluation_g = f.create_group('evaluation')
                inference_g = f.create_group('inference')
                tracking_g = f.create_group('tracking')
                
                for g in [evaluation_g, inference_g]:
                    g.create_dataset("NbDetection", (1,), maxshape=(None,), dtype="i", chunks=True)
                    g.create_dataset("Detections", (0, self.detection_size), maxshape=(None, self.detection_size), dtype="float32", chunks=True)
                    g.create_dataset("AppearanceFeatures", (0, self.reid_size,), maxshape=(None, self.reid_size,), dtype="float32", chunks=True)
                    
                tracking_g.create_dataset("DetectionsToTrack", (0,), maxshape=(None,),dtype="i", chunks=True)
                tracking_g.create_dataset("TrackToDetection", (0,), maxshape=(None,),dtype="i", chunks=True)
                
    def write_detections(self, detections_list, reid_features, source_type):
        """
        Give a list where each element correspond to the detections in a single frame 
        this function append the detection at the end of the HDF5 file
        """
                  
        assert source_type == "evaluation" or source_type == "inference"
        
        nb_frame = len(detections_list)
        nb_det = [len(det) for det in detections_list]
        nb_det_total = sum(nb_det)
        
        log.debug(f"Writing {nb_det_total} detections in {nb_frame} frames to {self.file_path}")

        with h5py.File(self.file_path, 'a') as f:
            #Get dataset
            nbDetdSet = f[f"{source_type}/NbDetection"]
            DetectionsSet = f[f"{source_type}/Detections"]
            AppearanceFeaturesSet = f[f"{source_type}/AppearanceFeatures"]
            
            #Extend size to accomodate new data
            nbDetdSet.resize((len(nbDetdSet) + nb_frame,))
            DetectionsSet.resize((len(DetectionsSet) + nb_det_total, self.detection_size))
            if reid_features is not None:
                AppearanceFeaturesSet.resize((len(AppearanceFeaturesSet) + nb_det_total, self.reid_size))
            
            #Add data to respective dataset
            prev_cumsum = 0 if len(nbDetdSet) == nb_frame else nbDetdSet[-nb_frame-1]
            nbDetdSet[-nb_frame:] = np.cumsum(np.array(nb_det)) + prev_cumsum
            
            if nb_det_total > 0:
                DetectionsSet[-nb_det_total:] = np.vstack(flatten(detections_list))
            
            if reid_features is not None and nb_det_total > 0:
                AppearanceFeaturesSet[-nb_det_total:] = np.vstack(flatten(reid_features))
                
    def get_detection(self, frame_id, source_type, get_reid=False):
        """
        Given a frame id retrieve the corresponding detection and reid
        """
        
        assert source_type == "evaluation" or source_type == "inference"
        
        log.debug(f"Retrieving detection for frame {frame_id} from {self.file_path}")
        
        with h5py.File(self.file_path, 'a', swmr=True) as f:
            nbDetdSet = f[f"{source_type}/NbDetection"]
            DetectionsSet = f[f"{source_type}/Detections"]
            AppearanceFeaturesSet = f[f"{source_type}/AppearanceFeatures"]
            
            assert frame_id >= 0 and frame_id < len(nbDetdSet)
            
            det_start = nbDetdSet[frame_id]
            det_end = nbDetdSet[frame_id+1]
            
            dets = DetectionsSet[det_start:det_end]
            
            if get_reid:
                reids = AppearanceFeaturesSet[det_start:det_end]
            else:
                reids = None
        
        return dets, reids
            
    def get_nb_frames(self, source_type):

        assert source_type == "evaluation" or source_type == "inference"
        with h5py.File(self.file_path, 'a', swmr=True) as f:
            nbDetdSet = f[f"{source_type}/NbDetection"]
            nb_frames = len(nbDetdSet) - 1
            
        return nb_frames
