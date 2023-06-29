import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import cv2
import numpy as np
import configparser

from pathlib import Path

from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string

from utils.multiview_utils import MultiviewVids
from detection.configs.arguments import get_train_config_dict
from detection.inference import start_inference

if __name__ == '__main__':
    log.info("2 - Starting inference on validation sequence -")

    conf_dict =  get_config_dict()
    log.debug(f'conf_dict: {dict_to_string(conf_dict["training"])}')
    root_dir = Path(conf_dict["main"]["data_root"]) / "2-training/"

    mvv = MultiviewVids() 

    train_conf = get_train_config_dict(conf_dict["training"]["train_config_file"])
    train_conf["training"]["ROOT_PATH"] = str(root_dir)
    #setting path to data and output directories     
    
    train_conf["data_conf"]["pipeline_conf"] = conf_dict
    train_conf["data_conf"]["pipeline_mvv"] = mvv
    train_conf["data_conf"]["mode"] = "evaluation"

    log.debug(dict_to_string(train_conf))

    start_inference(train_conf)

    log.info("2 - Validation completed -")
    