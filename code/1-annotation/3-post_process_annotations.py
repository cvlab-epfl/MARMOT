import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import cv2
import numpy as np

from pathlib import Path

from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string
from utils.visualization import save_video_mp4, visualize_gt_cv2

from utils.multiview_utils import MultiviewVids

if __name__ == '__main__':
    log.info("3 - Annotation post processing -")

    conf_dict =  get_config_dict()
    log.debug(f'conf_dict: {dict_to_string(conf_dict["annotation"])}')

    root_dir = Path(conf_dict["main"]["data_root"]) / "1-annotation/"
    nb_train_sample = conf_dict["annotation"]["num_train_sample"]
    val_seq_length = conf_dict["annotation"]["val_seq_length"]

    mvv = MultiviewVids()

    multiview_gt = mvv.load_gt("val", undisort=False)

    for i, cam in enumerate(mvv):
        frame_list_path = list((root_dir / "val" / cam.name).glob('*.png'))
        # remove hidden files
        frame_list_path = [frame for frame in frame_list_path if not frame.name.startswith(".")]

        frame_list_indices = [int(frame.stem.split("_")[1]) 
                              for frame in frame_list_path]

        frame_list = [cv2.imread(str(frame)) for frame in frame_list_path]
        frame_list = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                      for frame in frame_list]

        gt_list = [[ann.feet for ann in multiview_gt[i][frame_index]] 
                   for frame_index in frame_list_indices]
        

        frame_with_gt = [visualize_gt_cv2(frame, gt_points=gt, roi=None) 
                         for frame, gt in zip(frame_list, gt_list)]

        save_video_mp4(frame_with_gt, 
                       str(root_dir / "visualisation" / f"{cam.name}_val_ann"))
        log.debug(f"Annotation video saved at: "
                  f"{root_dir / 'visualisation' / f'{cam.name}_val_ann'}")

    log.info("3 - Annotation postprocessing completed -")
    