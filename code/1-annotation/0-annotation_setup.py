import sys

if "../" not in sys.path:
	sys.path.insert(0, "../")

from pathlib import Path

from configs.arguments import get_config_dict
from utils.log_utils import log, dict_to_string
from utils.visualization import save_video_mp4
from utils.multiview_utils import MultiviewVids

from ann_utils import sample_frames, sample_sequence

if __name__ == '__main__':
    log.info("0 - Annotation setup starting -")

    conf_dict =  get_config_dict()
    log.debug(f'conf_dict: {dict_to_string(conf_dict["annotation"])}')

    root_dir = Path(conf_dict["main"]["data_root"]) / "1-annotation/"
    nb_train_sample = conf_dict["annotation"]["num_train_sample"]
    val_seq_length = conf_dict["annotation"]["val_seq_length"]
    val_seq_start = conf_dict["annotation"]["val_seq_start"]
    sample_multiplier_val = 3
    sample_multiplier_train = 3

    mvv = MultiviewVids()
    max_id = mvv.get_max_frame_id()

    if sample_multiplier_train * nb_train_sample > max_id:
        raise ValueError("Not enough training frames to sample from")
    
    log.info(f"Extracting validation sequence of len {val_seq_length}")
    #Extract validation sequence from videosp
    if val_seq_start == -1:
        seq_frame_indices = list(range(val_seq_start, val_seq_start+val_seq_length))
    else:
        seq_frame_indices = sample_sequence(mvv, val_seq_length, sample_multiplier=10)

    for cam in mvv:
        frame_list = cam.extract(seq_frame_indices)
        cam.save(root_dir / "val" / cam.name, frame_list)
        save_video_mp4(frame_list, 
                       str(root_dir / "visualisation" / f"{cam.name}_val"), save_framerate = 1)


    log.info(f"Extracting {nb_train_sample} training frames")
    max_id = mvv.get_max_frame_id() - 1

    #Extract training frame from videos
    selected_frames_indices = []
    for cam in mvv:
        selected_frames_indices.extend(
             sample_frames(cam, nb_train_sample//mvv.get_nb_cams(), 
                           sample_multiplier=sample_multiplier_train, 
                           exclude=seq_frame_indices, max_idx=max_id))
        
    log.info(f"Saving {selected_frames_indices} training frames"
                f" for each camera")  
    for cam in mvv:
        frames = cam.extract(selected_frames_indices)
        log.info(f"Saving {len(frames)} frames for {cam.name}")
        cam.save(root_dir / "train" / cam.name, 
                 frames)


    log.info("0 - Annotation setup completed -")