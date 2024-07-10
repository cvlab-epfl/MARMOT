import os
import json
from pathlib import Path


remote_volume = Path('/run/user/1000/gvfs/') / 'smb-share:server=drobo6.local,share=data' / 'Studies/DM-Karlsruhe-2023/TrackingExterne/2023-09-20-mercredi/'
# smb-share\:server\=drobo6.local\,share\=data/Studies/DM-Karlsruhe-2023/TrackingExterne/2023-09-20-mercredi/')
destination_folder = Path('/home/marmot/MARMOT/data_source2/raw_data/footage')
remote_360_volume = Path('/run/user/1000/gvfs/') / 'smb-share:server=drobo6.local,share=data' / 'Studies/DM-Karlsruhe-2023/Photos/360/Dm-Store.mp4'
calib_path = Path('/home/marmot/MARMOT/data/0-calibration/calibs/')
checkerboard_path = Path('/home/marmot/MARMOT/data_source2/raw_data/calibration')

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

for vid_id, vid_path in enumerate(sorted(remote_volume.iterdir())):
    # remote_vid_path = remote_volume / vid_path
    cam_folder = destination_folder / f'cam{vid_id + 1}'
    cam_folder.mkdir(parents=True, exist_ok=True)
    cam_file = cam_folder / vid_path.name
    if cam_file.is_symlink():
            cam_file.unlink()
    cam_file.symlink_to(vid_path)

    calib["view_id"] = vid_id + 1
    json_dict = json.dumps(calib, indent = 4)
    with open(calib_path / f"cam{vid_id + 1}.json", 'w') as f:
        f.write(json_dict)
        

    
cam_folder = destination_folder / f'cam{360}'
cam_folder.mkdir(parents=True, exist_ok=True)
cam_file = cam_folder / '360.mp4'

if cam_file.is_symlink():
            cam_file.unlink()
cam_file.symlink_to(remote_360_volume)

checkerboard_path.mkdir(parents=True, exist_ok=True)


