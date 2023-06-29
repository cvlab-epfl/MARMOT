import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from ctypes import c_bool
from multiprocessing import Queue
from pathlib import Path

import numpy as np
import torch

from detection.dataset import factory as data_factory
from detection.dataset.utils import is_in_frame

from detection.loss import factory as loss_factory

from detection.misc import detection, geometry
from detection.misc.log_utils import DictMeter, batch_inf_logging, dict_to_string
from detection.misc.metric import compute_mot_metric_from_det, save_detection_for_evaluation
from detection.misc.utils import listdict_to_dictlist, check_for_existing_checkpoint
from detection.model import factory as model_factory

from utils.storage_utils import InferenceHDF5
from utils.log_utils import log

warnings.filterwarnings("ignore", category=UserWarning)

class InferenceEngine():
    def __init__(self, model, conf, visualization=False):
        super(InferenceEngine, self).__init__()

        self.model = model
        self.conf = conf

        self.nb_views = len(conf["data_conf"]["view_ids"])

        self.use_nms = True
        self.nms_kernel_size = 3

        self.visualization = visualization

        if self.conf["data_conf"]["mode"] == "evaluation":
            self.storage_path = Path(conf["training"]["ROOT_PATH"]).parent / "2-training" / f'{conf["main"]["name"]}_evaluation.hdf5'
        elif self.conf["data_conf"]["mode"] == "inference":
            self.storage_path = Path(conf["training"]["ROOT_PATH"]).parent / "3-inference" / f'{conf["main"]["name"]}_inference.hdf5'

        self.detection_size = 2

        self.view_feat_size, self.multi_feat_size = model.multiview_model.get_reid_size()
        self.reid_size =  self.view_feat_size*self.nb_views+self.multi_feat_size
        self.storage = InferenceHDF5(self.storage_path, self.detection_size, self.reid_size)

        self.detection_to_evaluate = conf["training"]["detection_to_evaluate"]#["pred_0"]

        log.info("Loading Data ...")
        end = time.time()
        self.nb_frame_already_processed = self.storage.get_nb_frames()
        if self.nb_frame_already_processed > 0:
            log.warning(f"Found {self.nb_frame_already_processed} already processed frames in the storage. Will skip them.")

        _, self.val_loader = data_factory.get_dataloader(self.conf["data_conf"], skip_init=self.nb_frame_already_processed)
        self.val_loader = self.val_loader[0]
        log.info(f"Data loaded in {time.time() - end} s")

        self.total_nb_frames = len(self.val_loader)

    def run(self, epoch):
        
        self.epoch = epoch
        self.result_dicts = []
        self.stats_meter = DictMeter()

        end = time.time()
        for f, input_data in enumerate(self.val_loader):
            input_data = input_data.to(self.conf["device"])
            
            data_time = time.time() - end

            with torch.no_grad():
                output_data = self.model(input_data, return_features=True)

                #put all the output in cpu to free gpu memory for the remaining of validation
                output_data = output_data.to("cpu")
                input_data = input_data.to("cpu")
                
                # Extract detected point
                processed_results, output_data = self.post_process_heatmap(input_data, output_data)
                #Compute detection and count metric if groundtruth is available

                self.store_step_dict(input_data, output_data, processed_results)
            
            batch_time = time.time() - end

            epoch_stats_dict = {**output_data["time_stats"], "batch_time":batch_time, "data_time":data_time, "criterion_time":0, "optim_time":0}
            self.stats_meter.update(epoch_stats_dict)
            
            if f % self.conf["main"]["print_frequency"] == 0 or f == (self.total_nb_frames - 1):
                batch_inf_logging(self.epoch, f, self.total_nb_frames, self.stats_meter)
            
            if f % self.conf["main"]["save_frequency"] == 0 or f == (self.total_nb_frames - 1):
                result_dict = listdict_to_dictlist(self.result_dicts)
                self.storage.write_detections(result_dict["pred_0_points"], result_dict["pred_0_scores"], result_dict["pred_0_appearance"])
                self.storage.write_heatmaps({det_k:result_dict[det_k] for det_k in self.detection_to_evaluate})
                self.result_dicts = []
            end = time.time()
            #When we have accumulated max_tracklet_lenght step dict or reach the en dof dataset we push the step dict to the tracker process


    def post_process_heatmap(self, input_data, output_data):
        
        #post process detection heatmap from self.detection_to_evaluate list
        processed_results = dict()

        #Set prediction outside of ROI to zero
        for det_k in self.detection_to_evaluate:
            if det_k.split("_")[0] != "framepred":
                output_data["pred"][det_k] = output_data["pred"][det_k] * input_data["ROI_mask"]
            else:
                v_id = int(det_k.split('_')[2][1])
                output_data["pred"][det_k] = output_data["pred"][det_k] * input_data["ROI_image"][:,v_id]


        #Decode heatmap to get detection and corresponding appearance feature
        for det_k in self.detection_to_evaluate:
            scores_flow, pred_point_flow = detection.decode_heatmap(output_data["pred"][det_k], self.nms_kernel_size, self.use_nms, threshold="auto")

            #project groundplane detection to each view
            if det_k.split("_")[0] != "framepred":                
                #add view apperance
                appearance_feat = np.zeros((pred_point_flow.shape[0], self.reid_size))# len(pred_point_flow)
                for v, view_id in enumerate(self.conf["data_conf"]["view_ids"]):
                    pred_point_v = geometry.project_points(pred_point_flow, input_data["homography"][0,v])
                    mask_visible = np.array([is_in_frame(point[[1,0]], self.conf["data_conf"]["hm_image_size"]) for point in pred_point_v])
                    # pred_point_v = pred_point_v[mask_visible]
                    
                    # (np.array([[50,20],[50,100],[10,100],[30,20]]), [True, True, True, True])
                    for i, (point, visible) in enumerate(zip(pred_point_v, mask_visible)):
                        #extract appearance feature for each detection
                        if not visible or input_data["ROI_image"][0, v, int(point[1]), int(point[0])] == 0:
                           continue          
                        appearance_feat[i,  v*self.view_feat_size:(v+1)*self.view_feat_size] = output_data["pred"]["view_feat"][v, :, int(point[1]), int(point[0])]

                for i, point in enumerate(pred_point_flow):
                    appearance_feat[i,  -self.multi_feat_size:] = output_data["pred"]["scene_feat"][0, :, int(point[1]), int(point[0])]
            else:
                appearance_feat = None
                #Add scene apperance feature

            #extract appearance feature for each detection

            processed_results[det_k+"_points"] = pred_point_flow
            processed_results[det_k+"_scores"] = scores_flow
            processed_results[det_k+"_appearance"] = appearance_feat


        return processed_results, output_data

    def store_step_dict(self, input_data, output_data, processed_results):
        """
        Store combination of input and prediction to generate tracker, metrics, and visualiztion
        We assume batchsize is 1 and only take the first element of the batch and the first view
        """
        step_dict = {}

        #Adding detection
        for det_k in self.detection_to_evaluate:
             step_dict[det_k] = output_data["pred"][det_k]
             step_dict[det_k+"_points"] = processed_results[det_k+"_points"]
             step_dict[det_k+"_scores"] = processed_results[det_k+"_scores"]
             step_dict[det_k+"_appearance"] = processed_results[det_k+"_appearance"]

        self.result_dicts.append(step_dict)



def start_inference(config):

    config["device"] = torch.device('cuda' if torch.cuda.is_available() and config["main"]["device"] == "cuda" else 'cpu') 
    log.info(f"Device: {config['device']}")
    
   
    best_checkpoint = check_for_existing_checkpoint(config["training"]["ROOT_PATH"], config["main"]["name"], get_best=True) # "model_335")#
    epoch = best_checkpoint['epoch']

    #print information about the best checkpoint
    log.debug(f"Best checkpoint found: {best_checkpoint['conf']['main']['name']} at epoch {best_checkpoint['epoch']}")
    # log.debug(f"Metrics at best checkpoint: {best_checkpoint['epoch_stats']}")

    metric_to_print = config["training"]["metric_to_print"]

    metrics = [best_checkpoint['epoch_stats']['val'][metric] for metric in metric_to_print]
    max_char_len = [max(len(metrn), len(f'{metrc:.3f}')) for metrn, metrc in zip(metric_to_print, metrics)]
    str_metric_lgd = f"Metric {'  '.join([f'{metr:<{padding}}' for metr, padding in zip(metric_to_print, max_char_len)])}"
    str_metric = f"Metric {'  '.join([f'{metric:<{padding}.3f}' for metric, padding in zip(metrics, max_char_len)])}"
    log.info("\t" + str_metric_lgd)
    log.info("\t" + str_metric)

    if best_checkpoint is None:
        log.error("No best checkpoint found, the model has not been trained yet")
        return
    
    log.info("Initializing model ...")
    end = time.time()

    model = model_factory.pipelineFactory(config["model_conf"], config["data_conf"])
    model.load_state_dict(best_checkpoint["state_dict"])
    model.to(config["device"])

    log.info(f"Model initialized in {time.time() - end} s")
    
    log.info(f"{f' Beginning validation for epoch {epoch} ':*^150}")
    infengine = InferenceEngine(model, config, visualization=config["data_conf"]["mode"]=="evaluation")
    valid_results = infengine.run(epoch)