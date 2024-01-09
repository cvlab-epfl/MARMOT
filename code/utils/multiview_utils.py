import os
import cv2
import re
import json
import copy
import sys
# import bisect
# import concurrent.futures
# from tqdm import tqdm
# import concurrent.futures

# from joblib import Parallel, delayed

from pathlib import Path
from collections import defaultdict, Counter, namedtuple
from typing import Any, List, Dict, Tuple, Union, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from skspatial.objects import Vector, Line, Plane, Point

from detection.misc import geometry
from detection.dataset.utils import Annotations
from utils.io_utils import load_json, write_json, is_media_file
from utils.log_utils import log, dict_to_string
from configs.arguments import get_config_dict



Calibration = namedtuple('Calibration', 
                         ['K', 'K_new', 'R', 'T', 
                          'dist', 'view_id', 'ROI', 'bounding_box', 'size'],
                          defaults=(None, None, None, None, 
                                    None, None, None, None, None))

def json_numpy_decoder(dct):
    """
    Custom JSON Decoder for numpy data types.
    Converts lists to numpy arrays after
    deserializing from JSON.
    """
    for k, v in dct.items():
        print(v)
        if isinstance(v, list):
            if all(isinstance(item, list) for item in v):
                dct[k] = np.array([np.array(item, dtype=np.float64) 
                                   for item in v])
            else:
                dct[k] = np.array(v, dtype=np.float64)
    return dct

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder for numpy data types.
    Converts numpy arrays to lists before 
    serializing to JSON.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BaseCamera:
    """
    Camera Parent Class that implements functionality 
    to extract frames from a video.

    Arguments:
    ----------
    name: str
        Name of the camera.
    data_root: str, optional
        Path to the root directory of the dataset.
    """
    def __init__(self, name: str, data_root:Optional[Path] = None):
        self.name = name
        self.data_root = (data_root or 
                          Path(get_config_dict().get('main', 
                                                {}).get('data_root', 
                                                        '/root/data')))
        self.omni_tag = get_config_dict().get('calibration', 
                                              {}).get('omni_tag', '360')
        
        self.env_footage = self.data_root / 'raw_data' / 'footage' / self.name

        self.calib_footage = self.data_root / 'raw_data' / 'calibration' / self.name

        self.video_dict = {'calibration': {}, 'footage': {}}
        self.num_frames = 0
        self.is_omni = self.omni_tag in self.name
        self.first_frame = 0
        self.index_videos()
        
        first_frame_path = self.data_root / '0-calibration' / 'first_frame.json'
        self.set_first_frame(first_frame_path)
        
        self.frame_ids = []

        self.frame_size = None

        self.config = get_config_dict()

    def index_videos(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """
        Indexes videos associated with the camera and 
        creates a dictionary of video names and frame ranges.

        Returns:
        --------
        video_dict: dict
            A dictionary of video names and frame ranges, 
                    organized by type (calibration or footage).
        """
        num_frames = 0
        for video_type, video_dir in [('calibration', self.calib_footage), 
                                      ('footage', self.env_footage)]:
            prev_end = 0
            if video_dir.is_dir():
                for vid in sorted(os.listdir(video_dir)):
                    vid_path = str(video_dir / vid)
                    if is_media_file(vid):
                        log.debug(f"Indexing {video_type} video {vid}...")
                        frames_vid = int(cv2.VideoCapture(
                            vid_path).get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        if video_type == 'footage':
                            num_frames += frames_vid
                        self.video_dict[video_type][vid] = {"start": prev_end, 
                                                            "end": prev_end 
                                                            + frames_vid - 1}
                        prev_end += frames_vid

        self.num_frames = num_frames
        log.debug(f"Indexed {self.num_frames} frames for camera {self.name}")
        return self.video_dict
    
    def __len__(self, *args: Any, **kwds: Any) -> Any:
        """
        Returns the number of frames in the camera.
        """
        return self.num_frames
    
    def set_first_frame(self, path:Optional[Path] = None) -> int:
        """
        Sets the first frame of the camera.

        Arguments:
        ----------
        path: str, optional
            Path to a json file containing the first frame of the camera.

        Returns:
        --------
        first_frame: int
            The first frame of the camera.
        """
        if path is None:
            if self.is_omni:
                first_frame = 0
            else:
                min_key = min(self.video_dict['footage'], 
                              key=lambda k: 
                              self.video_dict['footage'][k]['start'])
                first_frame = self.video_dict['footage'][min_key]['start']
        else:
            try:
                with open(path, 'r') as f:
                    dictionary = json.load(f)
                if self.name not in dictionary:
                    first_frame = 0
                else:
                    first_frame = dictionary[self.name]
            except FileNotFoundError:
                first_frame = 0

        self.first_frame = first_frame
        self.max_frame = self.num_frames - 1
        self.num_frames -= first_frame
        
        
        return first_frame
    
    def _extract_frame(self, video_capture: cv2.VideoCapture, 
                       frame_number: int, max_attempts: int = 50) -> np.ndarray:
        """Extracts a frame from a video capture object.

        Arguments:
        ----------
        video_capture : cv2.VideoCapture
            The video capture object.
        frame_number : int
            The frame number to extract.
        max_attempts : int, optional
            The maximum number of attempts to load the frame (default is 10).

        Returns:
        --------
        frame : np.ndarray
            The extracted frame.
        """
        #With grab to read a frame video seems to be one indexed, we shift the frame number by 1
        frame_number += 1

        if (frame_number < 0 or 
            frame_number > video_capture.get(cv2.CAP_PROP_FRAME_COUNT)):
            raise ValueError(f"Invalid frame number {frame_number} for video "
                            f"with {video_capture.get(cv2.CAP_PROP_FRAME_COUNT)}"
                            " frames")

        if max_attempts <= 0:
            raise ValueError("max_attempts must be a positive integer")
        
        # video_capture.set(cv2.CAP_PROP_POS_AVI_RATIO, frame_number / video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        curr_frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)

        if curr_frame_number > frame_number or curr_frame_number < 0:
            log.debug("Resetting video capture")
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 1)
            curr_frame_number = 1

        for i in range(int(frame_number - curr_frame_number)):
            for attempt in range(max_attempts):
                ret = video_capture.grab()
                if ret:
                    break
            if not ret:
                log.warning(f"Failed to grab frame {i} ")

        # assert video_capture.get(cv2.CAP_PROP_POS_FRAMES) == frame_number

        ret, frame = video_capture.retrieve()
        if ret:
            return frame
        
        log.warning(f"Failed to load frame {frame_number} "
                    f"after {max_attempts} attempts "
                    "returning blank frame")
        return np.zeros((self.calibration.size[0], 
                         self.calibration.size[1], 3), 
                         dtype=np.uint8)
    
    def extract(self, frame_ids: List[int], 
                mode: str='footage', rgb=False) -> List[np.ndarray]:
        """
        Extracts frames from the specified video files. Frame ids are shifted
        by the first frame of the camera.

        Args:
            frame_ids: A list of frame IDs to extract.
            mode: The mode to extract frames from.
                    'footage' - Extract frames from the footage videos.
                    'calibration' - Extract frames from the calibration videos.

        Returns:
            A list of extracted frames.
        """
        # Discard frame IDs that are out of range
        frame_ids = [frame_id + self.first_frame for frame_id in frame_ids]

        valid_frame_ids = [frame_id for frame_id in frame_ids 
                           if 0 <= frame_id <= self.max_frame]
        
        discarded_ids = set(frame_ids) - set(valid_frame_ids)
        
        if len(discarded_ids) > 0:
            log.warning(f"Discarding frame IDs {discarded_ids} "
                        "that are out of range.")

        # Sort the frame IDs and keep track of the original order
        sorted_indices = np.argsort(valid_frame_ids)
        unsorted_indices = np.argsort(sorted_indices)
        sorted_frame_ids = np.array(valid_frame_ids)[sorted_indices]

        # Extract frames from each video file
        extracted_frames = []
        current_frame_index = 0

        for video_file in self.video_dict[mode]:
            start_frame = self.video_dict[mode][video_file]['start']
            end_frame = self.video_dict[mode][video_file]["end"]

            # Find the first frame that is within the current video
            while (current_frame_index < len(sorted_frame_ids) and 
                   (sorted_frame_ids[current_frame_index] < start_frame)):
                current_frame_index += 1
            # Extract frames from the current video
            video = cv2.VideoCapture(str(self.data_root / 'raw_data' / mode 
                                     / self.name / video_file))
            
            for idx in range(current_frame_index, len(sorted_frame_ids)):
                # print(f"idx: {idx}, len(sorted_frame_ids): {len(sorted_frame_ids)}")
                if sorted_frame_ids[idx] > end_frame:
                    break

                # Print progress every 100 frames
                if idx % 1 == 0:
                    log.spam(f"Extracting frame {sorted_frame_ids[idx]} "
                             f"from {self.name}...")

                shifted_id = sorted_frame_ids[idx] - start_frame

                if idx % 1 == 0:
                    log.spam(f"Extracting frame {shifted_id} "
                                f"from {video_file}... ")
                
                frame = self._extract_frame(video, shifted_id)
                if rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                extracted_frames.append(frame)

                # Compare the current frame to last frame in extracted_frames
                # If the frames are the same, remove the current frame
                if (len(extracted_frames) > 2 and 
                    np.array_equal(extracted_frames[-1], extracted_frames[-2])):
                    log.warning(f"Frame {shifted_id} same as the previous frame...")

            current_frame_index = idx
            

            video.release()

        # Warn if the number of extracted frames 
        # doesn't match the number of requested frames
        if len(extracted_frames) != len(valid_frame_ids):
            log.warning(f"Expected to extract {len(valid_frame_ids)} frames but"
                         f" extracted {len(extracted_frames)} frames instead.")

        # Restore the original order of the extracted frames

        try:
            extracted_frames = [extracted_frames[index] for index 
                                in unsorted_indices[:len(extracted_frames)]]
        except IndexError:
            log.warning("Failed to extract a frame, saving successful frames.")
            extracted_frames = [extracted_frames[index] for index 
                                in unsorted_indices 
                                if index < len(extracted_frames)]

        self.frames = extracted_frames
        self.frame_ids = valid_frame_ids
        return extracted_frames

    def save(self, output:str, frames:List[np.ndarray] = []) -> None:
    
        """Saves frames to output directory
        Arguments:
        ----------
            output(str): path to output directory
        """
        log.debug(f"Saving frames to {output}...")
        if not os.path.isdir(output):
            log.info(f"Creating output directory {output}")
            os.makedirs(output)

        if len(frames) == 0:
            frames = self.frames
        
        for i, frame in enumerate(frames):
            index = i
            file_path =  Path(output) / f'{self.name}_{str(index)}.png' 
            success = cv2.imwrite(str(file_path), frame)
            if not success:
                log.warning(f"Failed to save frame {index} to {file_path}")
    
    def get_frame_size(self):
        """
        Returns the frame size of the camera's images.
        """
        if self.frame_size is None:
            self.extract([0])
            self.frame_size = np.array(self.frames[0].shape[:2])
            log.debug(f"Cam {self.name} Frame size: {self.frame_size}")

        return self.frame_size 
    
    def get_max_frame_id(self) -> int:
        """
        Returns the maximum frame ID of the camera.
        """
        return self.num_frames - 1


class Camera(BaseCamera):
    """
    Calibrated Camera that stores intrinsics and extrinsics. Allows for 
    undistortion of frames and conversion between camera and world coordinates.
    """
    def __init__(self, name:str, newest:bool=True):
        super().__init__(name)
        self.calibration = Calibration(view_id = self.name.split('m')[1])

        new_calib_path = (self.data_root / '0-calibration' / 'calibs' 
                          / f'{self.name}_new.json')
        old_calib_path = (self.data_root / '0-calibration' / 'calibs'
                            / f'{self.name}.json')

        if newest and new_calib_path.is_file():
                self.calib_path = new_calib_path
        else:
            self.calib_path = old_calib_path

        self.calibration = None
        self.calibration = self.get_calib()

        self.padding_percent = self.config["calibration"]['rectangle_padding']
        


    def is_calibrated(self) -> bool:
        """
        Checks if the camera is calibrated.
        If the camera is fully calibrated: returns True
        If ROI and Bounding Box are missing: returns True
        If extrinsics are missing: returns False
        If Intrinsics are missing: returns False


        Returns:
        --------
            is_calibrated(bool): whether the camera is calibrated
        """
        if self.calibration.K is None:
            return False
        elif self.calibration.R is None:
            return False
        elif self.calibration.ROI is None:
            return True
        else:
            return True
        
    def undistort(self, frames:List[np.ndarray]=[], 
                  output:Optional[Path]=None, use_k_new=None
                  ) -> List[np.ndarray]:
        """
        Undistorts all the frames in self.frames and saves them
        to output/undistorted if output is not None

        Arguments:
        ----------
            frames(list): list of frames to undistort
            output(str): path to output directory

        Returns:
        --------
            undistorted(list): list of undistorted frames
        """
        if use_k_new is None:
            use_k_new = self.config["calibration"]["use_alpha_undistort"]
            
        if output is not None:
            output = Path(output) / 'undistorted' / self.name
            output.mkdir(parents=True, exist_ok=True)
        
        frames_to_undistort = []

        if len(frames) == 0:
            frames_to_undistort = self.frames
            frame_ids = self.frame_ids
        else:
            frames_to_undistort = frames
            frame_ids = range(len(frames))
        
        undistorted = []

        for i, frame in enumerate(frames_to_undistort):
            if use_k_new:
                undistorted_frame = cv2.undistort(frame, 
                                                  np.array(self.calibration.K), 
                                                  np.array(self.calibration.dist), 
                                                  None, 
                                                  np.array(self.calibration.K_new))
            else:
                undistorted_frame = cv2.undistort(frame, 
                                                  np.array(self.calibration.K), 
                                                  np.array(self.calibration.dist),
                                                  None)

            undistorted.append(undistorted_frame)

            if output is not None:
                file_path = output / f'{self.name}_{frame_ids[i]}.png'               
                cv2.imwrite(str(file_path), undistorted_frame)

        return undistorted

    def convert_to_camera_frame(self, points:np.ndarray
                                ) -> Tuple[np.ndarray, List[bool]]:
        """
        Takes in a list of points in world frame and 
        converts them to the camera frame.

        Arguments:
        ----------  
            points(list): list of points in world frame

        Returns:
        --------
            points(list): list of points in camera frame
            behind_camera(list): list of booleans indicating 
                        whether the point is behind the camera
        """
        assert points.shape[1] in (2, 3), ("Points must be a list of 3D points "
                                           "or 2D points with z=1")

        if points.shape[1] == 2:
            points = np.hstack((points, np.ones((points.shape[0], 1))))

        points = points.T
        points = self.calibration.R @ points + self.calibration.T.reshape(3, 1)
        points = self.calibration.K @ points
        
        behind_camera = points[2] < 0
        points = points / points[2]
        points = points[:2].T

        return points, behind_camera

    def convert_to_world_frame(self, points:np.ndarray) -> np.ndarray:
        """
        Takes in a list of points in camera frame and 
        converts them to world frame.

        Arguments:
        ----------
            points(list): list of points in camera frame

        Returns:
        --------
            points(list): list of points in world frame
        """
        assert points.shape[1] in (2, 3), ("Points must be a list of 3D points" 
                                           "or 2D points with z=1")

        if points.shape[1] == 2:
            points = np.hstack((points, np.ones((points.shape[0], 1))))

        points = np.linalg.inv(self.calibration.R) @ (points 
                                                      - self.calibration.T).T


        return points.T
    
    def from_image(self, points:np.ndarray) -> np.ndarray:
        """
        Takes in a list of points in image coordinates and 
        converts them to the camera frame.

        Arguments:
        ----------
            points(list): list of points in image coordinates

        Returns:
        --------
            points(list): list of points in camera frame
        """

        points = np.array(points)
        
        if points.shape[1] == 2:
            points = np.hstack((points, np.ones((points.shape[0], 1))))
        assert points.shape[1] == 3, ("Points must be a list of 3D points "
                                      "or 2D points with z=1")

        points = points.T
        points = np.linalg.inv(self.calibration.K) @ points

        return points.T
    
    def get_position(self) -> np.ndarray:
        """
        Returns the position of the camera in world frame.

        Returns:
        --------
            center(np.ndarray): center of the camera in world frame
        """
        return self.convert_to_world_frame(np.array([[0, 0, 0]])).reshape(3,)

    def save_calibration(self, calibration:Calibration=Calibration(), 
                         calib_path:Optional[Path]=None) -> None:
        """
        Saves the calibration of the camera to a json file.

        Arguments:
        ----------
            calibration(Calibration): calibration data
            calib_path(str): path to calibration file
        """
        # if all the calibration parameters are None, 
        # then use the current calibration
        new_calib = {key: value if value is not None 
                     else self.calibration._asdict()[key]
                     for key, value in calibration._asdict().items() 
                     if key in self.calibration._asdict()}
        
        calibration = Calibration(**new_calib)


        if calib_path is None:
            calib_path = self.calib_path
        try:
            calib_path.parent.mkdir(parents=True, exist_ok=True)
            with open(calib_path, 'w') as f:
                json.dump(calibration._asdict(), f, 
                          indent=2, cls = NumpyEncoder)
        except:
            raise ValueError("Unable to write JSON {}".format(calib_path))

    def get_calib(self, calib_path:Optional[Path]=None) -> Calibration:
        """
        Loads the calibration of the camera from a json file.
        
        Arguments:
        ----------
            calib_path(str): path to calibration file
        """
        if self.calibration is not None:
            return self.calibration
        
        if calib_path is None:
            calib_path = self.calib_path

        try:
            with open(calib_path) as f:    
                data = json.load(f)
                if data['K'] is not None:
                    data['K'] = np.array(data['K']).astype(np.float32)
                if data['R'] is not None:
                    data['R'] = np.array(data['R']).astype(np.float32)
                if data['T'] is not None:
                    data['T'] = np.array(data['T']).astype(np.float32)
                if data['K_new'] is not None:
                    data['K_new'] = np.array(data['K_new']).astype(np.float32)
                if data['dist'] is not None:
                    data['dist'] = np.array(data['dist']).astype(np.float32)
                if data['view_id'] is not None:
                    data['view_id'] = int(data['view_id'])
                if data['size'] is not None:
                    data['size'] = tuple(data['size'])
                if data['bounding_box'] is not None:
                    data['bounding_box'] = data['bounding_box']
                if data['ROI'] is not None:
                    data['ROI'] = data['ROI']
                # load the calibration from the json file
                self.calibration = Calibration(**data)
                # extract test frame and set size
                frame = self.extract([0])[0]
                self.calibration = self.calibration._replace(
                                size=(frame.shape[0], frame.shape[1]))

        except Exception as e:
            log.warning(f"Unable to read JSON {calib_path}, "
                        "initialising with default values")
            log.warning(e)
            self.calibration = Calibration()

        return self.calibration
    
    def set_calib(self, calibration: Calibration, save: bool = False) -> None:
        """
        Sets the calibration of the camera.

        Arguments:
        ----------
        calibration(Calibration): calibration data
        save(bool): whether to save the calibration to a json file
        """
        if calibration.view_id is None:
            calibration = calibration._replace(view_id=self.name.split('m')[1])

        new_calib = {key: value if value is not None 
                     else self.calibration._asdict()[key]
                     for key, value in calibration._asdict().items() 
                     if key in self.calibration._asdict()}

        self.calibration = Calibration(**new_calib)

        if save:
            self.save_calibration(self.calibration)

    def calib_from_reconstruction(self, path:Optional[Path] = None, 
                                  reconstruction: dict = {}, 
                                  ) -> Calibration:
        """
        Calculates the calibration of the camera from a reconstruction file.

        Arguments:
        ----------
        path(str): path to reconstruction file
        reconstruction(dict): reconstruction data

        Returns:
        --------
        calibration(Calibration): calibration data
        """
        if not path and not reconstruction:
            raise ValueError("Either path or reconstruction must be specified.")

        if path:
            if not isinstance(path, Path):
                raise TypeError("Expected reconstruction path to be a string.")

            if not path.is_file():
                raise FileNotFoundError(f"The specified reconstruction file "
                                        f"'{path}' does not exist.")

            try:
                reconstruction = load_json(str(path))[0]
            except (json.decoder.JSONDecodeError, IndexError):
                raise ValueError(f"The specified reconstruction file "
                                 f"'{reconstruction}' is not valid.")

        shots = reconstruction.get('shots', {})
        rot, trans, n = np.zeros(3), np.zeros(3), 0

        for shot, shot_data in shots.items():
            if not isinstance(shot_data, dict):
                log.warning(f"Invalid shot data for shot '{shot}', skipping.")
                continue

            if not shot.startswith(f'{self.name}_'):
                continue

            if not all(key in shot_data for key in ('rotation', 'translation')):
                log.warning(f"Missing rotation or translation data for shot "
                             f"'{shot}', skipping.")
                continue

            try:
                rot += np.array(shot_data["rotation"])
                trans += np.array(shot_data["translation"])
                n += 1
            except ValueError:
                log.warning(f"Invalid rotation or translation data for shot "
                            f"'{shot}', skipping.")

        if n == 0:
            log.warning(f"No valid shots found for camera '{self.name}', "
                        "skipping.")
        else:
            self.set_calib(Calibration(
                R=np.array(R.from_rotvec(rot/n).as_matrix()), 
                T=np.array(trans/n), view_id=self.calibration.view_id, 
                K=self.calibration.K, 
                K_new=self.calibration.K_new, 
                dist=self.calibration.dist,
                size = self.calibration.size))

        return self.calibration
    
    
    def get_ground_plane_homography(self, 
            input_img_size = (1080, 1920),
            output_img_size = (1080, 1920),
            bounding_box = ((0., 0.), (50., 50.), 0),
            padding_percent=0.
            ) -> np.ndarray:
        """
        Computes the homography that projects ground plane onto image plane.
        Scales the homography such that the image fills out the bounding box.

        Arguments:
        ----------
        input_img_size(Tuple[int, int]): input image size 
                                            (height, width) in pixels
        output_img_size(Tuple[int, int]): output image size 
                                            (height, width) in pixels
        bounding_box(Tuple[Tuple, Tuple, float]): bounding box 
                                            ((x, y), (width, height), angle)

        Returns:
        --------
        H(np.ndarray): homography matrix
        """

        # Create a new rotation-translation matrix RT by combining 
        # the rotation matrix R and translation vector T
        RT = np.zeros((3,3))
        RT[:,:2] = self.calibration.R[:,:2]
        RT[:,2] = self.calibration.T.squeeze()

        # Scale the bounding box by the output image size
        scale = max(output_img_size) / max(bounding_box[1])
        
        bounding_box = ((bounding_box[0][0]*scale, bounding_box[0][1]*scale), 
                        (bounding_box[1][0]*scale, bounding_box[1][1]*scale), 
                        bounding_box[2])


        # if longest sides of output image and bounding box don't align, rotate the bounding box
        if output_img_size[1] > output_img_size[0] and bounding_box[1][1] > bounding_box[1][0]:
            print("Rotating bounding box counter clockwise 90 degrees")
            bounding_box = ((bounding_box[0][1], bounding_box[0][0]), 
                            (bounding_box[1][1], bounding_box[1][0]), 
                            bounding_box[2] + 90)

        if output_img_size[0] > output_img_size[1] and bounding_box[1][0] > bounding_box[1][1]:
            print("Rotating bounding box clockwise 90 degrees")
            bounding_box = ((bounding_box[0][1], bounding_box[0][0]), 
                            (bounding_box[1][1], bounding_box[1][0]), 
                            bounding_box[2] - 90)


        # Rotate by rotation around the z axis
        rot = R.from_rotvec(np.radians(bounding_box[2])*np.array([0, 0, 1])).as_matrix()

        # Compute the translation to center the output image bbox center with padding
        cx, cy = bounding_box[0][1], bounding_box[0][0]

        if padding_percent != 0:
            padding_x = int(output_img_size[1] / padding_percent / 100)
            padding_y = int(output_img_size[0] / padding_percent / 100)
        else:
            padding_x = 0
            padding_y = 0
        tx = int((output_img_size[1]-1) / 2) + cx
        ty = int((output_img_size[0]-1) / 2) + cy

        # Create Ki with padding
        Ki = np.eye(3)
        Ki[0,0] = scale / (1 + padding_percent / 100)
        Ki[1,1] = scale / (1 + padding_percent / 100)
        Ki[0,2] = tx + padding_x
        Ki[1,2] = ty + padding_y

        # Create conversion matrix from input_img_size to native size
        factorx = input_img_size[1] / (self.calibration.size[1])
        factory = input_img_size[0] / (self.calibration.size[0])

        # New camera matrix to account for image size
        K_new = self.calibration.K.copy()
        K_new[0,0] = self.calibration.K[0,0] * factorx
        K_new[1,1] = self.calibration.K[1,1] * factory
        K_new[0,2] = self.calibration.K[0,2] * factorx
        K_new[1,2] = self.calibration.K[1,2] * factory

        H = K_new @ RT @ rot @ np.linalg.inv(Ki) # 

        return H


class MultiviewVids():
    """Multiview video class
    Extracts frames from multiview videos
    Loads or calculates calibration data"""
    def __init__(self, config:dict = {}, newest:bool = True):
        """
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary, by default {}
        newest : bool, optional
            If True, use the newest calibration data, by default True
        """
        self.config = get_config_dict()
        data = (Path(self.config["main"]["data_root"]) / "raw_data" / "footage")
        calib_root = (Path(self.config["main"]["data_root"]) 
                      / "0-calibration" / "calibs")
        self.cam_names = self.get_cam_names(str(calib_root))
        
        self.label_root = (Path(self.config["main"]["data_root"]) 
                           / "1-annotation" / "labels")

        log.info(f"Cameras: {self.cam_names}")
        self.omni_tag = self.config['calibration']['omni_tag']

        self.cams = []
        self.name_to_cam = {}
        for cam in self.cam_names:
            camera = Camera(cam, newest=newest)
            if camera.is_calibrated():
                log.info(f"Loading calibration data for camera '{cam}'")
                self.cams.append(camera)
                self.name_to_cam[cam] = self.cams[-1]

        self.max_frames = self.get_max_frame_id()
        
    def __iter__(self):
        return self.cams.__iter__()
    
    def __getitem__(self, key):
        return self.cams[key]

    def get_nb_cams(self) -> int:
        """Returns the number of cameras
        """
        return len(self.cams)
    
    def get_max_frame_id(self) -> int:
        """Returns the maximum frame id
        """
        self.max_frames = np.min([cam.get_max_frame_id() for cam in self.cams])

        return self.max_frames
    
    def get_frame_size(self):
        """Returns the frame size
        """
        return self.cams[0].calibration.size
        frame_sizes = [cam.get_frame_size() for cam in self.cams]
        #check that all frame sizes are the same
        if not all((x == frame_sizes[0]).all() for x in frame_sizes):
            log.warning("Frame sizes are not the same for all cameras,"
                        "this may cause issues.")

        return self.cams[0].get_frame_size()
    
    def extract_mv(self, frame_ids:List[int], undistort:bool = False
                   ) -> Dict[str, List[np.ndarray]]:
        """Multiview extraction
        Extracts frames for each of the cameras in self.cams
        Shifts frame ids to match the first frame of each camera
        Arguments:
        ----------
            frame_ids(list): list of frame ids

        Returns:
        --------
            frames(dictionary): dictionary of frames. Each entry is named
                    after the camera that the frames derive from and contains
                    a list of the frames for that camera, synced by first frame.
        
        """
        cam_frames = {}

        max_frame_id = self.get_max_frame_id()
        # truncate frame ids to max frame id
        frame_ids = [min(frame_id, max_frame_id) for frame_id in frame_ids]

        for cam in self.cams:
            # cam_frame_ids = [frame_ids[i] + cam.first_frame 
            #                  for i in range(len(frame_ids))]
            cam_frames[cam.name] = cam.extract(frame_ids)
            if undistort:
                cam_frames[cam.name] = cam.undistort(cam_frames[cam.name])
        return cam_frames

    def get_cam_names(self, folder: Optional[str]= None) -> list:
        """
        Returns a list of all the camera directories in the folder.
        Cam directories should contain 'cam'.

        Arguments:
        ----------
        folder(str): path to the parent directory of the camera files
        """
        if folder is None:
            return [cam.name for cam in self.cams]
        else:
            pattern = re.compile(r"cam\d+")

            cams = [cam for cam in os.listdir(folder) if pattern.match(cam) 
                    and self.config['calibration']['omni_tag'] not in cam]

            cam_names = [cam.split('.')[0].split('_')[0] for cam in cams]

            # remove duplicates
            cam_names = list(set(cam_names))
            cam_names.sort()

            return cam_names
        
    def get_cameras(self) -> list:
        """Returns list of Cameras"""
        return self.cams

    def get_cam_by_name(self, name):
        """Returns list of Cameras"""
        if name in self.name_to_cam:
            return self.name_to_cam[name]
        else:
            log.error(f"Camera {name} doesn't exist returning None")
            return None
    
    def set_first_frames(self, first_frame:dict) -> None:
        """Sets first frame for all cameras
        Arguments:
        ----------
            first_frame(int): first frame
        """
        for cam in self.cams:
            try:
                cam.first_frame = first_frame[cam.name]
            except KeyError:
                log.warning(f"First frame for {cam.name} not found, "
                            "leaving as 0")
                pass

    def get_calibrations(self) -> List[Calibration]:
        """Returns Calibrations for each camera
        
        Arguments:
        ----------
            data(str): path to calibration data
            
        Returns:
        ----------
            list[Calibration]: list of calibrations
        """
        calibrations = []
        for cam in self.cams:
            calibrations.append(cam.get_calib())
            
        self.calibrations = calibrations

        return calibrations
    
    def get_homographies(self, input_size:tuple = (1080, 1920), 
                         output_size:tuple = (1080, 1920)
                         ) -> List[np.ndarray]:
        """Returns homographies for each camera
        
        Arguments:
        ----------
            input_size(tuple): size of input image (height, width)
            output_size(tuple): size of output image (height, width)
            
        Returns:
        ----------
            list[np.ndarray]: list of homographies
        """
        bbox, _ = self.get_bounding_box()
        homographies = []
        for cam in self.cams:
            homographies.append(cam.get_ground_plane_homography(
                input_img_size = input_size, 
                output_img_size = output_size, 
                bounding_box = bbox))
            
        return homographies
    
    def get_bounding_box(self, reconstruction:dict = {}, 
                         ground_plane:Plane = Plane(point=Point([0, 0, 0]), 
                                                    normal=Vector([0, 0, 1]))
                        ) -> Tuple[Tuple[
                                    Tuple[float, float],
                                    Tuple[float, float], 
                                    float], 
                                    np.ndarray]:
        """Calculates the bounding box of the cameras in the reconstruction
        based on the region of interest of each camera and the ground plane.

        Args:
            reconstruction (dict, optional): Reconstruction dictionary. 
                                                Defaults to {}.
            ground_plane (Plane, optional): Ground plane. 
                    Defaults to Plane(point=Point([0, 0, 0]), 
                    normal=Vector([0, 0, 1])).

        Returns:
            bounding_box: bounding box of the cameras
            roi_array: points in region of interest
        """
        roi_list = []
        for camera in self.cams:
            if reconstruction != {}:
                camera.calibration = camera.calib_from_reconstruction(
                    reconstruction=reconstruction)
                
            cam_center = camera.convert_to_world_frame(
                np.array([[0,0,0]])).reshape(3,)
            
            for point in camera.calibration.ROI:
                roi_list.append(ground_plane.intersect_line(Line.from_points(
                    Point(camera.convert_to_world_frame(
                    camera.from_image(np.array([point]))).reshape(3,)), 
                    Point(cam_center))))
                
        roi_array = np.array(roi_list)
        bounding_box = cv2.minAreaRect((roi_array[:, :2]).astype(np.float32))

        return bounding_box, roi_array
    
    def get_view_ROIs(self) -> List[np.ndarray]:
        """Returns view ROIs for each camera"""
        view_ROIs = []
        for cam in self.cams:
            view_ROIs.append(np.array(cam.calibration.ROI))
            
        return view_ROIs

    def load_gt(self, set_name:str, undisort:bool = True) -> dict:
        """Loads ground truth annotations for a given set
        Arguments:
        ----------
            set_name(str): name of the set
            undisort(bool): whether to undisort the annotations
        Returns:
        ----------
            multiview_gt(dict): dictionary of annotations
        """

        gt_folder = self.label_root / set_name
        anns_path_list = list(gt_folder.glob('*.json'))

        if len(anns_path_list) == 0:
            log.warning(f"No annotations found in {gt_folder}")
            return {key:defaultdict(list) for key in range(len(self.cams))}
        
        unique_annotators_dict = Counter([path.stem.split("_")[0] 
                                          for path in anns_path_list])
        if len(unique_annotators_dict.keys()) > 1:
            biggest_annotator = list(unique_annotators_dict.keys())[
                list(unique_annotators_dict.values()).index(
                max(unique_annotators_dict.values()))]
            
            log.warning(f"Annotations from multiple annotators found in "
                        f"{gt_folder}. Using annotations from "
                        f"{biggest_annotator}")

            anns_path_list = [path for path in anns_path_list 
                              if biggest_annotator == path.stem.split("_")[0]]

        
        multiview_gt = {k:defaultdict(list) for k in range(len(self.cams))}

        for ann_path in anns_path_list:
            ann_data = load_json(str(ann_path))[1:]

            frame_id = int(ann_path.stem.split("_")[1])
            log.spam(f"Loading annotations for frame {frame_id}...")

            for view_id in range(len(self.cams)):
                calib = self.cams[view_id].get_calib()

                # log.debug(f"Loading annotations for view {view_id}... with calib {calib}")

                for ann in ann_data:
                    world_point = np.array([[ann[3]], [ann[4]], [ann[5]]])
                    if undisort:
                        feet_reproj = geometry.project_world_to_camera(
                            world_point, calib.K, calib.R, calib.T)
                    else:
                        feet_reproj = geometry.project_world_to_camera(
                            world_point, calib.K_new, calib.R, 
                            calib.T, calib.dist)
                    
                    person_id = int(ann[1])
                    multiview_gt[view_id][frame_id].append(
                        Annotations(bbox=None, head=None,  
                                    feet=feet_reproj, height=180, 
                                    id=person_id, frame=frame_id, 
                                    view=view_id))

        return multiview_gt
    
    def update_calib(self, translation:np.ndarray = np.array([0.,0.,0.]), 
                     rotation:float = 0, scale:float = 1.)->List[Calibration]:
        """Updates the extrinsics stored in calib to
        account for a translation and a rotation.

        Arguments:
            translation (np.ndarray) : translation to be applied to calib
            rotation (float)         : rotation to be applied to calib
                                        in degrees about z axis
            scale (float)            : scale to be applied to calib

        Returns:
        ----------
            Calibration: updated calibration
        """
        r = R.from_euler('z', rotation, degrees=True).as_matrix()
        T_n = np.hstack((translation, 0))

        for cam in self.cams:
            new_T = cam.calibration.T * scale
            new_R = cam.calibration.R @ r
            new_T = cam.calibration.T + (new_R @ T_n) # check sign (new_R @ T_n)
            cam.set_calib(Calibration(
                K=cam.calibration.K, R=new_R, 
                T=new_T, dist=cam.calibration.dist, 
                view_id=cam.calibration.view_id, 
                K_new = cam.calibration.K_new,
                ROI = cam.calibration.ROI,
                bounding_box = cam.calibration.bounding_box))

        self.calibrations = self.get_calibrations()
        return self.calibrations



