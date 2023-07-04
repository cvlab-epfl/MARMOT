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
                hm_group = f.create_group('heatmaps')
                evaluation_g.create_dataset("NbDetection", (1,), maxshape=(None,), dtype="i", chunks=True)
                evaluation_g.create_dataset("Detections", (0, self.detection_size), maxshape=(None, self.detection_size), dtype="float32", chunks=True)
                evaluation_g.create_dataset("DetectionsConf", (0,), maxshape=(None,), dtype="float32", chunks=True)
                evaluation_g.create_dataset("AppearanceFeatures", (0, self.reid_size,), maxshape=(None, self.reid_size,), dtype="float32", chunks=True)

                
    def write_detections(self, detections_list, detection_conf, reid_features):
        """
        Give a list where each element correspond to the detections in a single frame 
        this function append the detection at the end of the HDF5 file
        """
        
        nb_frame = len(detections_list)
        nb_det = [len(det) for det in detections_list]
        nb_det_total = sum(nb_det)
        
        log.debug(f"Writing {nb_det_total} detections in {nb_frame} frames to {self.file_path}")

        with h5py.File(self.file_path, 'a') as f:
            #Get dataset
            nbDetdSet = f[f"evaluation/NbDetection"]
            DetectionsSet = f[f"evaluation/Detections"]
            DetectionsConfSet = f[f"evaluation/DetectionsConf"]
            AppearanceFeaturesSet = f[f"evaluation/AppearanceFeatures"]
            
            #Extend size to accomodate new data
            nbDetdSet.resize((len(nbDetdSet) + nb_frame,))
            DetectionsSet.resize((len(DetectionsSet) + nb_det_total, self.detection_size))
            DetectionsConfSet.resize((len(DetectionsConfSet) + nb_det_total,))

            if reid_features is not None:
                AppearanceFeaturesSet.resize((len(AppearanceFeaturesSet) + nb_det_total, self.reid_size))
            
            #Add data to respective dataset
            prev_cumsum = 0 if len(nbDetdSet) == nb_frame else nbDetdSet[-nb_frame-1]
            nbDetdSet[-nb_frame:] = np.cumsum(np.array(nb_det)) + prev_cumsum
            
            if nb_det_total > 0:
                DetectionsSet[-nb_det_total:] = np.vstack(flatten(detections_list))
                DetectionsConfSet[-nb_det_total:] = np.hstack(flatten(detection_conf))
            
            if reid_features is not None and nb_det_total > 0:
                AppearanceFeaturesSet[-nb_det_total:] = np.vstack(flatten(reid_features))

                
    def get_detection(self, frame_id, get_reid=False):
        """
        Given a frame id retrieve the corresponding detection and reid
        """
        
        log.spam(f"Retrieving detection for frame {frame_id} from {self.file_path}")
        
        with h5py.File(self.file_path, 'a', swmr=True) as f:
            nbDetdSet = f[f"evaluation/NbDetection"]
            DetectionsSet = f[f"evaluation/Detections"]
            DetectionsConfSet = f[f"evaluation/DetectionsConf"]
            AppearanceFeaturesSet = f[f"evaluation/AppearanceFeatures"]
            
            assert frame_id >= 0 and frame_id < len(nbDetdSet)
            
            det_start = nbDetdSet[frame_id]
            det_end = nbDetdSet[frame_id+1]
            
            dets = DetectionsSet[det_start:det_end]
            confidence = DetectionsConfSet[det_start:det_end]

            
            if get_reid:
                reids = AppearanceFeaturesSet[det_start:det_end]
            else:
                reids = None
        
        return dets, confidence, reids
    
    def write_heatmaps(self, maps_dict):
        """
        Write the heatmap to the HDF5 file
        """
        
        log.debug(f"Writing following heatmaps {list(maps_dict.keys())} to {self.file_path}")

        for det_type, map_list in maps_dict.items():
            nb_maps = len(map_list)
            maps_size = map_list[0].squeeze().shape

            with h5py.File(self.file_path, 'a') as f:
                #check if dataset exists
                if f"heatmaps/{det_type}" not in f:
                    f.create_dataset(f"heatmaps/{det_type}", (1, *maps_size), maxshape=(None, *maps_size), dtype="float32", chunks=True)
                    f.create_dataset(f"heatmaps/Nb{det_type}", (1,), maxshape=(None,), dtype="i", chunks=True)

                #Get dataset
                nbDetdSet = f[f"heatmaps/Nb{det_type}"]
                MapsSet = f[f"heatmaps/{det_type}"]

                nbDetdSet.resize((len(nbDetdSet) + nb_maps,))
                MapsSet.resize((len(MapsSet) + nb_maps, *maps_size))

                nbDetdSet[0] = nbDetdSet[0] + nb_maps
                MapsSet[-nb_maps:] = np.vstack(map_list).squeeze()

    def get_heatmaps(self, det_type, start_frame, end_frame, agg="avg"):

        assert agg == "avg" or agg == "max"

        with h5py.File(self.file_path, 'a') as f:
            #Get dataset
            nbDetdSet = f[f"heatmaps/Nb{det_type}"]
            MapsSet = f[f"heatmaps/{det_type}"]

            assert start_frame >= 0 and start_frame < len(MapsSet)
            assert end_frame >= 0 and end_frame < len(MapsSet)
            
            res = np.zeros(MapsSet[0].shape)

            for i in range(start_frame, end_frame):

                if agg == "avg":
                    res += MapsSet[i] / len(MapsSet)
                elif agg == "max":
                    res = np.maximum(res, MapsSet[i])

            return res
            
    def get_nb_frames(self):
        with h5py.File(self.file_path, 'a', swmr=True) as f:
            nbDetdSet = f[f"evaluation/NbDetection"]
            nb_frames = len(nbDetdSet) - 1
            
        return nb_frames
    

class TrackingHDF5:
    """
    A class to store inference results in a HDF5 file.
    """
    def __init__(self, path, detection_size=2):
        super(TrackingHDF5, self).__init__()
        
        if type(path) == str:
            path = Path(path)
            
        self.file_path = path
        self.detection_size = detection_size
        
        self.create_file()
            
    def create_file(self):
        if not(self.file_path.is_file()):
            log.debug(f"Creating HDF5 file {self.file_path}")
            self.file_path.parents[0].mkdir(parents=True, exist_ok=True)
            with h5py.File(self.file_path, 'w') as f:
                tracking_g = f.create_group('tracking')

                tracking_g.create_dataset("TrackToID", (0,), maxshape=(None,), dtype="i", chunks=True)
                tracking_g.create_dataset("TrackToFrameWindow", (0,2), maxshape=(None,2), dtype="i", chunks=True)
                tracking_g.create_dataset("NbDetection", (1,), maxshape=(None,), dtype="i", chunks=True)
                tracking_g.create_dataset("Detections", (0, self.detection_size), maxshape=(None, self.detection_size), dtype="float32", chunks=True)
                tracking_g.create_dataset("DetectionsFrameId", (0,), maxshape=(None,), dtype="float32", chunks=True)


                
    def write_tracks(self, tracks_list):
        """
        Give a list where each element correspond to the detections in a single frame 
        this function append the detection at the end of the HDF5 file
        """

        source_type = "tracking"      
        
        nb_track = len(tracks_list)
        nb_det = [len(track) for track in tracks_list]
        nb_det_total = sum(nb_det)
        
        log.debug(f"Writing {nb_det_total} detections within a number ot track {nb_track} to {self.file_path}")

        with h5py.File(self.file_path, 'a') as f:
            #Get dataset
            nbDetdSet = f[f"{source_type}/NbDetection"]
            DetectionsSet = f[f"{source_type}/Detections"]
            DetectionsFrameIdSet = f[f"{source_type}/DetectionsFrameId"]
            track_to_id = f[f"{source_type}/TrackToID"]
            track_to_frame_window = f[f"{source_type}/TrackToFrameWindow"]

            #Extend size to accomodate new data
            nbDetdSet.resize((len(nbDetdSet) + nb_track,))
            DetectionsSet.resize((len(DetectionsSet) + nb_det_total, self.detection_size))
            DetectionsFrameIdSet.resize((len(DetectionsFrameIdSet) + nb_det_total,))


            track_to_id.resize((len(track_to_id) + nb_track,))
            track_to_frame_window.resize((len(track_to_frame_window) + nb_track, 2))

            #Add data to respective dataset
            prev_cumsum = 0 if len(nbDetdSet) == nb_track else nbDetdSet[-nb_track-1]
            nbDetdSet[-nb_track:] = np.cumsum(np.array(nb_det)) + prev_cumsum
            
            if nb_det_total > 0:
                detections_list = [np.array([det["X"], det["Y"]]) for track in tracks_list for det in track]
                DetectionsSet[-nb_det_total:] = np.vstack(detections_list)

                detections_frame_id_list = np.array([det["FrameId"] for track in tracks_list for det in track])
                DetectionsFrameIdSet[-nb_det_total:] = detections_frame_id_list

                start_end = [[det["FrameId"] for det in track] for track in tracks_list]
                start_end = [np.array([min(track), max(track)]) for track in start_end]
                track_to_frame_window[-nb_track:] = np.vstack(start_end)

                id_track = np.array([track[0]["Id"] for track in tracks_list])
                track_to_id[-nb_track:] = id_track

    def get_tracks_as_df(self, window_start=None, window_end=None):
        track_as_list = []

        tracks, frame_ids = self.get_tracks(window_start, window_end)

        for person_id, (track, frame_id) in enumerate(zip(tracks, frame_ids)):
            for frame_id, detection in zip(frame_id, track):
                track_as_list.append({'FrameId':int(frame_id), 'Id':int(person_id), 'X':int(detection[0]), 'Y':int(detection[1])})

        return track_as_list

    def get_tracks(self, window_start=None, window_end=None):
        """
        Given a frame id retrieve the corresponding detection and reid
        """
        if window_start is not None and window_end is not None:
            assert window_start < window_end

        source_type = "tracking"
        
        log.debug(f"Retrieving detection for tracks in the frame window {window_start} to {window_end} from {self.file_path}")
        
        with h5py.File(self.file_path, 'a', swmr=True) as f:
            nbDetdSet = f[f"{source_type}/NbDetection"]
            DetectionsSet = f[f"{source_type}/Detections"]
            DetectionsFrameIdSet = f[f"{source_type}/DetectionsFrameId"]
            track_to_id = f[f"{source_type}/TrackToID"]
            track_to_frame_window = f[f"{source_type}/TrackToFrameWindow"]
            
            all_windows = track_to_frame_window[()]

            all_tracks = list()
            all_frame_ids = list()
            for track_id, window in enumerate(all_windows):
                if (window_start is not None and window[0] < window_start) or (window_end is not None and window[1] > window_end):
                    continue

                det_start = nbDetdSet[track_id]
                det_end = nbDetdSet[track_id+1]

                dets = DetectionsSet[det_start:det_end]
                frame_ids = DetectionsFrameIdSet[det_start:det_end]

                all_tracks.append(dets)
                all_frame_ids.append(frame_ids)

        return all_tracks, all_frame_ids
            
    def get_nb_tracks(self):
        source_type = "tracking"

        with h5py.File(self.file_path, 'a', swmr=True) as f:
            nbDetdSet = f[f"{source_type}/NbDetection"]
            nb_tracks = len(nbDetdSet) - 1
            
        return nb_tracks
