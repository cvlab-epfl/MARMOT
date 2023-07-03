import sys
import argparse
import configparser

from detection.configs.pathes import conf_path
from detection.configs.utils import read_yaml_file, convert_yaml_dict_to_arg_list, fill_dict_with_missing_value, aug_tuple
from detection.configs.utils import args_to_dict
from utils.log_utils import log, dict_to_string


parser = argparse.ArgumentParser(description='Process some integers.')

parser_train = parser.add_argument_group("training")
parser_data = parser.add_argument_group("dataset")
parser_model = parser.add_argument_group("model")
parser_loss = parser.add_argument_group("loss")


####### Configuration #######
parser.add_argument("-n", '--name', default="model", help='Name of the model')
parser.add_argument("-pf", "--print_frequency", dest="print_frequency", help="Number of element processed between print", type=int, default=2500)
parser.add_argument("-sf", "--save_frequency", dest="save_frequency", help="Number of element processed between saving inference result to database", type=int, default=100)
parser.add_argument("-dev", "--device", dest="device", help="select device to use either cpu or cuda", default="cuda")
parser.add_argument("-ch", "--nb_checkpoint", dest="nb_checkpoint", default=10, type=int, help="maximum number to checkpoint to save, once reach will start overwrite old checkpoint")
parser.add_argument("-l", "--log_lvl", dest="log_lvl", default="debug", choices=["debug", "spam", "verbose", "info", "warning", "error"], help='Set level of logger to get more or less verbose output')

####### Training #######
parser_train.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=0.001, help="Initialization of the learning rate")
parser_train.add_argument("-lrd", "--learning_rate_decrease", dest="lrd", nargs='+', type=float, default=[0.5, 20, 40, 60, 80, 100], help="List of epoch where the learning rate is decreased (multiplied by first arg of lrd)")
parser_train.add_argument("-dec", "--decay", dest="decay", type=float, default=5e-4, help="Adam weight decay")
parser_train.add_argument("-mepoch", '--max_epoch', dest="max_epoch", type=int, default=100, help="Max epoch")
parser_train.add_argument("-sstep", '--substep', dest="substep", type=int, default=1, help="Max epoch")
parser_train.add_argument("-gclip", '--gradient_clip_value', dest="gradient_clip_value", type=float, default=10000000000, help="Value over which the gradient is clipped to stabilize training and avoid gradient explosion")
parser_train.add_argument("-dtv", "--detection_to_evaluate", dest="detection_to_evaluate", nargs='+', type=str, default=["pred_0"], help="List of name of detection to evaluate at inference time")
parser_train.add_argument("-metp", "--metric_to_print", dest="metric_to_print", nargs='+', type=str, default=["moda_pred_0","recall_pred_0","precision_pred_0"], help="List of metric that will appeared in the validation log file every time a log is made")
parser_train.add_argument("-lott", "--loss_to_print", dest="loss_to_print", nargs='+', type=str, default=["loss_cont_det","loss_cont_rec","loss_cont_time_consistency"], help="List of losses that will appeared in the log file every time a log is made")


####### Data #######
parser_data.add_argument("-m", "--mode", dest="mode", default="training", choices=["training", "inference", "evaluation"], help='Dataset to use for Training')
parser_data.add_argument("-nbf", "--nb_frames", dest="nb_frames", type=int, default=1, help="Number of frame from the same scene to process at the same time")
parser_data.add_argument("-vid", "--view_ids", dest="view_ids", type=int, default=[0,1], nargs='*', help="Id of the views from the same scene to process at the same time start a 0")
parser_data.add_argument("-hmt", "--hm_type", dest="hm_type", default="center", choices=["density", "center", "constant"], help='type of methode use to generate groundtruth heatmaps')
parser_data.add_argument("-hms", "--hm_size", dest="hm_size", nargs="+", type=int, default=[180, 80], help="Expect pairs of integer coresponding to the size to resize the image before processing")
parser_data.add_argument("-hmr", "--hm_radius", dest="hm_radius", type=int, default=2,  help="The radius of the gaussian filter used to generate heatmpas")
parser_data.add_argument("-pr", "--pre_range", dest="pre_range", type=int, default=3,  help="size of the windows of frame a previous frame can be selected when building pairs")
parser_data.add_argument("-splt", "--split_proportion", dest="split_proportion", type=float, default=0.9, help="Train val split proportion the first split_proportion percent of the frames are used for training, the rest for validation")
parser_data.add_argument("-shft", "--shuffle_train", dest="shuffle_train", action="store_false", default=True, help="By default train set is shuffled, use arg to disable shuffling")
parser_data.add_argument("-bs", "--batch_size", dest="batch_size", type=int, default=1,  help="The size of the batches")
parser_data.add_argument("-dfps", "--desired_fps", dest="desired_fps", type=int, default=-1,  help="The desired framerate use to load the data, must be lower than the original framerate and a factor of it, if set to -1 it will default to the original framerate")
parser_data.add_argument("-cs", "--crop_size", dest="crop_size", nargs="+", type=int, default=[360, 640], help="Expect pairs of integer coresponding to the size to resize the image before processing") #[480, 832]
parser_data.add_argument("-nw", "--num_workers", dest="num_workers", type=int, default=2, help="Number of work to use in the train and val dataloader")
parser_data.add_argument("-mth", "--metric_threshold", dest="metric_threshold", type=float, default=2.5, help="Distance in groundplane pixel where detection are considered positive when computing metrics")
parser_data.add_argument("-aug", "--aug_train", dest="aug_train", action="store_true", default=False, help="Add various data augmentation to input frames during training")
parser_data.add_argument("-fi", "--frame_interval", dest="frame_interval", type=int, default=3, help="Interval between frame when sampling a video to make frame triplets")
parser_data.add_argument("-vaug", "--views_based_aug_list", dest="views_based_aug_list", type=aug_tuple, nargs='+', default=[(None, 1)], help="List of augmentation to sample as view based augmentation,  ('rcrop', 0.5) Augmentation tuple list is expected as AugType1,prob1 AugType2,prob2 ... AugTypeN,probN, the type and probability are separeted by a comma and each tuple is separated by a space")
parser_data.add_argument("-saug", "--scene_based_aug_list", dest="scene_based_aug_list", type=aug_tuple, nargs='+', default=[(None, 1)], help="List of augmentation to sample as scene based augmentation, ('raff', 0.5) Augmentation tuple list is expected as AugType1,prob1 AugType2,prob2 ... AugTypeN,probN, the type and probability are separeted by a comma and each tuple is separated by a space")


####### Model #######
parser_model.add_argument("-fgp", "--feature_ground_proj", dest="feature_ground_proj", action="store_true", default=False, help="By default ground projection is done on the input image, if set to true projection is moved in the feature space")
parser_model.add_argument("-ip", "--image_pred", dest="image_pred", action="store_true", default=False, help="By default the detection is only done in the groundplane, if true additional prediction and loss are applied in the image plaen in each view")

def get_train_config_dict(config_file):
    """
    Generate config dict from command line argument and config file if existing_conf_dict is not None, value are added to the existing dict if they are not already defined, 
    """

    if config_file is not None:
        yaml_dict = read_yaml_file(config_file)
        arg_list = convert_yaml_dict_to_arg_list(yaml_dict)
        args = parser.parse_args(arg_list)
    else:
        args = parser.parse_args("")

    args_dict = args_to_dict(parser, args)

    config = {"data_conf": args_dict["dataset"], "model_conf": args_dict["model"], "training": {**args_dict["training"], **conf_path}, "main":vars(args)}
    
        #
    config["data_conf"]["frame_input_size"] = config["data_conf"]["crop_size"]

    #if grounplane projection in middle homography take into account downscale of resnet by factor 8
    config["data_conf"]["homography_input_size"] = [x // 8  for x in config["data_conf"]["frame_input_size"]]
    config["data_conf"]["homography_output_size"] = config["data_conf"]["hm_size"]
    config["data_conf"]["hm_image_size"] = [x // 8  for x in config["data_conf"]["frame_input_size"]]

    return config


if __name__ == '__main__':
    conf_dict = get_train_config_dict()
