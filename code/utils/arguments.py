import argparse

def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

class Arguments():

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):

        #### CALIBRATOR ARGUMENTS ####
        self.parser.add_argument("--calibration_folder_final_output", "-cfo", 
                                 type=str, required=False, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/DATASET/calibration", 
                                 help="folder where xml files for intrinsics and extrinsics will be saved")
        
        self.parser.add_argument("--dataset_images_folder", "-if", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/DATASET/Image_subset", 
                                 help="folder where to store original and undistorted frames")
        
        self.parser.add_argument("--extracted_frames_folder", "-eff", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/DATASET/Image_subset/original", 
                                 help="folder where to store original frames in dataset folder")
        
        self.parser.add_argument("--undistorted_frames_folder", "-uff", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/DATASET/Image_subset/undistorted", 
                                 help="folder where to store undistorted frames in dataset folder")


        ### INTRINSICS CALIBRATION ARGUMENTS ###

        # if this is set to False then all below arguments are not considered
        self.parser.add_argument("--intrinsics_calibration", "-ic", 
                                 action="store_false", required=False)

        self.parser.add_argument("--calibration_folder_images", "-i", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/CALIBRATION_INPUT", 
                                 help="the folder where the original calibration videos are")
        
        self.parser.add_argument("--calibration_output_folder", "-co", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/CALIBRATION_OUT", 
                                 required=False, 
                                 help="folder for calibration output from opencv pipeline")
        
        self.parser.add_argument("--intrinsics_need_extract_frames", "-ine", 
                                 type=bool, default=False, required=False, 
                                 help="need to extract frames of calibration videos")
        
        self.parser.add_argument("--nc_to_calib", "-ncc", type=int, default=3, 
                                 required=False, 
                                 help="number of cameras we need to calibrate")

        # arguments required for compute_intrinsics.py
        self.parser.add_argument("--intrinsics_frame_rate", "-ifr", type=int, 
                                 default=0.9, required=False, 
                                 help="Frame rate for frame extraction in intrinsics calc.")
        
        self.parser.add_argument("--description", "-d", type=str, default="", 
                                 required=False, 
                                 help="Optional description to add to the output file.")
        
        self.parser.add_argument("--inner_corners_height", "-ich", type=int, 
                                 default=6, 
                                 help="Number of inner corners on the shortest edge of the checkerboard.")
        
        self.parser.add_argument("--inner_corners_width", "-icw", type=int, 
                                 default=9, 
                                 help="Number of inner corners on the longest edge of the checkerboard.")
        
        self.parser.add_argument("--square_sizes", "-s", type=int, default=40, 
                                 required=False, help="Size of the squares")
        
        self.parser.add_argument("--alpha", "-a", type=float, default=0.90, 
                                 required=False, 
                                 help="Parameter controlling the ammount of out-of-image pixels (\"black regions\") retained in the undistorted image.")
        
        self.parser.add_argument("--threads", "-t", type=int, default=4, 
                                 required=False)
        
        self.parser.add_argument("--force_monotonicity", "-fm", type=str2bool, 
                                 default=False, required=False, 
                                 help="Force monotonicity in the range defined by monotonic_range. To be used only in extreme cases.")
        
        self.parser.add_argument("--monotonic_range", "-mr", type=float, 
                                 default=-1, required=False, 
                                 help=("Value defining the range for the distortion must be monotonic. "
                                   "Typical value to try 1.3. Be careful: increasing this value may negatively perturb the distortion function."))
        
        self.parser.add_argument("--rational_model", "-rm", action="store_false", 
                                 required=False, 
                                 help="Use a camera model that is better suited for wider lenses.")
        
        self.parser.add_argument("--fix_principal_point", "-fpp", 
                                 action="store_true", required=False, 
                                 help="Fix the principal point either at the center of the image or as specified by intrisic guess.")
        
        self.parser.add_argument("--fix_aspect_ratio", "-far", 
                                 action="store_true", required=False) 
        
        self.parser.add_argument("--zero_tangent_dist", "-ztg", 
                                 action="store_true", required=False) 
        
        self.parser.add_argument("--criteria_eps", "-eps", type=float, 
                                 default=1e-5, required=False, 
                                 help="Precision criteria. A larger value can prevent overfitting and artifacts on the borders.")
        
        self.parser.add_argument("--fix_k1", "-k1", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--fix_k2", "-k2", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--fix_k3", "-k3", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--fix_k4", "-k4", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--fix_k5", "-k5", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--fix_k6", "-k6", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--intrinsic_guess", "-ig", type=str, 
                                 required=False, default="", 
                                 help="JSON file containing a initial guesses for the intrinsic matrix and distortion parameters.")
        
        self.parser.add_argument("--save_keypoints", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--load_keypoints", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--debug", action="store_true", required=False)


        ### OPENSFM ARGUMENTS ###

        ## DATA PREPARATION ARGUMENTS ##
        self.parser.add_argument("--preparation_input_folder", "-pi", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/VIDEO_FOLDER", 
                                 help="Folder containing #cam folders each one containing a video: input", 
                                 required=False)
        
        self.parser.add_argument("--nc_to_pose", "-ncp", type=int, default=4, 
                                 required=False, 
                                 help="number of cameras we need to estimate the pose")
        
        self.parser.add_argument("--extrinsic_calibration", "-ec", 
                                 action="store_false", required=False, 
                                 help="Cameras need their extrinsics to be computed")

        self.parser.add_argument("--need_to_add_images_to_dataset", "-nai", 
                                 action="store_true", required=False, 
                                 help="need to add images to the datset folder")
        
        self.parser.add_argument("--extrinsic_need_extract_frames", "-eef", 
                                 action="store_true", required=False)
        
        self.parser.add_argument("--extrinsic_need_undistort_frames", "-euf", 
                                 action="store_true", required=False)

        self.parser.add_argument("--extrinsics_frame_rate", "-efr", type=float, 
                                 default=0.1)

        self.parser.add_argument("--use_omnidir", "-om", action="store_true", 
                                 required=False)
        
        self.parser.add_argument("--omnidir_name", "-omn", type=str, 
                                 default="cam360")
        
        self.parser.add_argument("--extrinsics_frame_rate_omni", "-omfr", type=float, 
                                 default=1.0)

        ## OpenSFM ALGORITHM ARGUMENTS ##
        self.parser.add_argument("--frames_fisheye", "-ff", type=int, 
                                 default=5, 
                                 help="number of frames to use for each camera (not omni)")
        
        self.parser.add_argument("--frames_omnidir", "-fo", type=int, 
                                 default=-1, 
                                 help="number of frames to use for ommnidirectional")
        
        self.parser.add_argument("--openSFM_root", "-sfmroot", type=str, 
                                 default="/Users/sarno/Desktop/CALIBRATOR_TEST/openSFMroot", 
                                 help="root folder for opensfm, where data is stored")
        
        self.parser.add_argument("--need_to_create_opensfm_dir", "-nco", 
                                 action="store_true", help="need to create ")
        
        self.parser.add_argument("--openSFM_repo", "-sfmrepo", type=str, 
                                 default="/Users/sarno/Desktop/OpenSfM/", 
                                 help="repo folder for opensfm, where the files are stored")
        
        ## Projection ARGUMENTS ##
        self.parser.add_argument("--dataset", type=str, default="cvlab", 
                                 help="the dataset to use")
        

    def parse(self):
        self.args = self.parser.parse_args("")
        return self.args
    