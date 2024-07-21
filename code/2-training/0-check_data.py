import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import cv2
import numpy as np
import yaml 

from pathlib import Path

from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string

from utils.multiview_utils import MultiviewVids

def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')

yaml.add_representer(type(None), represent_none)

if __name__ == '__main__':
     log.info("0 - Checking training and validation data -")

     conf_dict =  get_config_dict()
     log.debug(f'conf_dict: {dict_to_string(conf_dict["training"])}')

     root_dir = Path(conf_dict["main"]["data_root"]) / "1-annotation/"

     mvv = MultiviewVids()
     nb_cam = mvv.get_nb_cams()
     base_cam = mvv.get_cameras()[0]

     log.info(f"Number of cameras found: {nb_cam}")



    
     for set_type in ["train", "val"]:
        multiview_gt = mvv.load_gt(set_type, undisort=False)

        nb_ann = max([len(ann) for ann in multiview_gt.values()]) if len(multiview_gt) > 0 else 0
        log.info(f"Number of annotations found for {set_type}: {nb_ann}")

        frame_list_path = list((root_dir / set_type / base_cam.name).glob('*.png'))

        log.info(f"Number of frame found for {set_type}: {len(frame_list_path)}")

        if nb_ann > len(frame_list_path):
             log.error(f"Number of annotations ({nb_ann}) is greater than number of frames ({len(frame_list_path)}) for {set_type}")

        if nb_ann < len(frame_list_path):
             log.warning(f"Number of annotations ({nb_ann}) is lower than number of frames ({len(frame_list_path)}) for {set_type}")
     
     log.info(f"Generating training configuration file")
     root_dir = Path(conf_dict["main"]["data_root"]) 
     root_code = Path(conf_dict["main"]["code_root"])

    #read default yaml file  
     with open(root_code.parent / "train_config.yaml", 'r') as stream:
           train_conf_def = yaml.safe_load(stream)

     log.spam(f'train_conf_default: {dict_to_string(train_conf_def)}')

     train_conf_def["training"]["--view_ids"] = [str(i) for i in range(nb_cam)]
     train_conf_def["training"]["--detection_to_evaluate"] = ["pred_0"] + [f"framepred_0_v{i}" for i in range(nb_cam)]

     log.spam(f'train_conf_ updated: {dict_to_string(train_conf_def)}')

     new_config_path = root_dir / "2-training/train_config.yaml"
     new_config_path.parent.mkdir(parents=True, exist_ok=True)
     #write new yaml file
     with open(root_dir / "2-training/train_config.yaml", 'w') as stream:
               yaml.dump(train_conf_def, stream)

     log.info(f"Training configuration file generated in {root_dir / '2-training/train_config.yaml'}")

     log.info("0 - Checking training and validation data completed -")
    