import os
import json
from pathlib import Path

# Path to the code directory
# BASEPATH = os.path.dirname(os.path.abspath(__file__)).split('code')[-2]
# CODEPATH = os.path.join(BASEPATH, 'code')
# DATAPATH = os.path.join(BASEPATH, 'data')
# sys.path.append(CODEPATH)

root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
calib_path = root_dir / 'data/0-calibration/calibs/'

data_folder = root_dir / 'data/raw_data'

checkerboard_path = data_folder / 'calibration'
footage_path = data_folder / 'footage'


calib = {
    "K":[
            [
                659.77,
                0.0,
                653.80
            ],
            [
                0.0,
                624.16,
                337.32
            ],
            [
                0.0,
                0.0,
                1.0
            ]
    ],
    "K_new":
        [
            [
            659.77,
            0.0,
            653.80
        ],
        [
            0.0,
            624.16,
            337.32
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ],
    "R":None,
    "T":None,
    "dist":[
        -0.16026,
        0.0262125,
        -0.00224888,
        -0.00259982,
        0.0
    ],
    "view_id":1,
    "ROI":None,
    "bounding_box":None,
    "size":None
}

calib_path.mkdir(parents=True, exist_ok=True)

for vid_id, vid_path in enumerate(sorted(footage_path.iterdir())):
    # remote_vid_path = remote_volume / vid_path
    # cam_folder = destination_folder / f'cam{vid_id + 1}'
    # cam_folder.mkdir(parents=True, exist_ok=True)
    # cam_file = cam_folder / vid_path.name
    # if cam_file.is_symlink():
    #         cam_file.unlink()
    # cam_file.symlink_to(vid_path)

    calib["view_id"] = vid_id + 1
    json_dict = json.dumps(calib, indent = 4)
    with open(calib_path / f"cam{vid_id + 1}.json", 'w') as f:
        f.write(json_dict)
        

    
# cam_folder = destination_folder / f'cam{360}'
# cam_folder.mkdir(parents=True, exist_ok=True)
# cam_file = cam_folder / '360.mp4'

# if cam_file.is_symlink():
#             cam_file.unlink()
# cam_file.symlink_to(remote_360_volume)

# checkerboard_path.mkdir(parents=True, exist_ok=True)


