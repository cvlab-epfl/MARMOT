import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")
        
import random 
import numpy as np

from pathlib import Path
from collections import defaultdict

from detection.misc.log_utils import dict_to_string
from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string

from utils.multiview_utils import MultiviewVids
from detection.configs.arguments import get_train_config_dict

from utils.storage_utils import InferenceHDF5
from utils.log_utils import log


conf_dict =  get_config_dict()
log.debug(f'conf_dict: {dict_to_string(conf_dict["annotation"])}')
root_dir = Path(conf_dict["main"]["data_root"]) / "2-training/"

mvv = MultiviewVids() 

train_conf = get_train_config_dict(conf_dict["training"]["train_config_file"])
train_conf["training"]["ROOT_PATH"] = str(root_dir)
#setting path to data and output directories     

train_conf["data_conf"]["pipeline_conf"] = conf_dict
train_conf["data_conf"]["pipeline_mvv"] = mvv
train_conf["data_conf"]["mode"] = "inference"

hm_size = np.array(train_conf["data_conf"]["hm_size"])
nb_views = mvv.get_nb_cams()

storage_path = Path(train_conf["training"]["ROOT_PATH"]).parent / "3-inference" / f'{train_conf["main"]["name"]}_inference.hdf5'
detection_size = 2

view_feat_size, multi_feat_size = 128, 32

reid_size =  view_feat_size*nb_views+multi_feat_size
storage = InferenceHDF5(storage_path, detection_size, reid_size)


nb_track = 50
nb_frames = 3000

pred = defaultdict(list)
appearance = defaultdict(list)


for track in range(nb_track):
    start_frame = random.randint(0, nb_frames)
    end_frame = random.randint(start_frame, nb_frames)

    base_apperance = np.random.rand(1, reid_size)
    base_pred = np.random.rand(1, 2) * hm_size[::-1]
    
    for frame in range(start_frame, end_frame):
        motion = (np.random.rand(1, 2) - 0.5) * hm_size[::-1] * 0.05
        apperance_noise = np.random.rand(1, reid_size) * 0.1

        base_pred = base_pred + motion
        # print(base_pred, hm_size)
        pred[frame].append(base_pred)
        appearance[frame].append(base_apperance + apperance_noise)

resul_pred = []
result_app = []

for frame in range(nb_frames):
    resul_pred.append(pred[frame])
    result_app.append(appearance[frame])


storage.write_detections(resul_pred, result_app, "inference")