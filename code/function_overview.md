# coordinate_utils:

- convert_axis_rotation(R: np.ndarray) -> np.ndarray:
- project_world_to_camera(world_point:np.ndarray, K1:np.ndarray, 
                            R1:np.ndarray, T1:np.ndarray) -> np.ndarray:
- project_floor(image:np.ndarray, origin:np.ndarray=np.array([0,0,0]), 
                  K1:np.ndarray=None, R1:np.ndarray=None, T1:np.ndarray=None, grid_size:int=10, 
                  grid_spacing:float=0.1, rotation:float=0.):

- convert_to_camera_frame(x: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
- get_ground_plane_homography(K:np.ndarray, R:np.ndarray, T:np.ndarray, 
                                groundplane_img_size:tuple, scale:tuple) -> np.ndarray:
- project_to_ground_plane_cv2(img:np.ndarray, H:np.ndarray, homography_output_size:tuple)->np.ndarray:
- add_axes(img:np.ndarray, calibration:Calibration, 
             origin:list[float]=[0,0,0], verbose:bool=False) -> None:


# io_utils:
- load_json(filename: str) -> dict:
- write_json(filename: str, data: dict) -> None:

# meta_data_utls:
- add_metadata(img_path:str, cam_name:str):
- get_cam_names(folder: str) -> list[str]:
- get_frame_from_file(frame_path:str) -> np.ndarray:


# calibration_utils:
- load_intrinsics(data_root = "/root/data")->dict:
- get_intrinsics(src:str=None, dest:str=None, args=None, force:bool=False)->dict:

# opensfm_utils:
- run_SfM(openSfM_repo: str=None, openSfM_data: str=None) -> int:
- create_openSFM_dir(openSfM_data: str, undistorted: bool=True, 
                       omni_tag: str='360') -> None:
- add_overrides(openSfM_data: str, undistorted: bool=True, omni_tag: str='360', 
                  focal_length: float=0.45) -> dict:
- add_OpenSfM_images(source:str, destination: str, undistorted: bool=True, 
                        omni_tag: str='360', num_frames_omni: int=None, 
                        num_frames: int = 2, verbose: bool = False) -> None:                                         
- update_reconstruction(reconstruction:dict=None, z:float=0) -> dict:
- extrinsics_from_reconstruction(reconstruction:str, omni_tag:str='360') -> dict: