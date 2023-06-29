import mot3d
import mot3d.weight_functions as wf
import mot3d.distance_functions as df
import numpy as np

from collections import Counter, defaultdict
from mot3d.utils import trajectory
from multiprocessing import Process, Value
from pathlib import Path

from utils.log_utils import log
from detection.misc.utils import flatten
from tracker.utils import Track, suppress_stdout


# python train.py -hmt center -splt 0.9 -dset retail_0 retail_1 -vis -bs 2 -flm prob -flagg sum -flc mse -clt mse -hmt constant -hmr 0 -rf 0 -fea r50 -lr 0.001 -mcon -fmt can -mva satt -nbv 6 -mtl 10 -pf 10 -of -cs 224 224 -n model_test_tracker2
class BaseTracker(Process):
    def __init__(self, conf, data_queue, result_queue=None, generate_viz=True):
        super(BaseTracker, self).__init__(daemon=True)
        self.conf = conf

        self.data_queue = data_queue
        self.result_queue = result_queue

        self.metric_threshold = 2.5

        self.time_id = 1
        self.next_time_id = 0
        self.det_type = "det"
        self.motion_dir = "b"

        # prediction name f"{self.det_type}_{self.time_id}{self.motion_dir}_points"
        # offset name f"offset_{self.time_id}_{self.next_time_id}{self.motion_dir}"

        self.groundplane_images = list()
        self.rois = list()
        self.gt_points = list()

        self.tracker_type = None

        self.all_tracks = list()
        self.all_scene_ids = list()
        self.all_homography = list()
        self.all_seq_starting_frame = list()

        self.prev_scene_id = -1
        self.current_frame_id = 0

        self.metrics = list()

        self.reset()

    def reset(self):
        self.tracklet_list = list()
        self.groundtruth_tracks = dict()
        self.pred_tracks = list()
    
    def run(self):
        log.spam(f"Starting tracker process {self.tracker_type} using {self.det_type}_{self.time_id}{self.motion_dir}_points and motion offset_{self.time_id}_{self.next_time_id}{self.motion_dir}")

        self.start_tracker_loop()

        if self.result_queue is not None:
            all_track = flatten(self.all_tracks)
            all_track = [track.get_track_as_list_of_dict() for track in all_track]
                              
            self.result_queue.put(all_track)
        
        log.debug("Tracker process finished")

    def start_tracker_loop(self):
        index_change = -1 

        #we loop until we get a -1 from the queue
        while True:
            if index_change == -1:
                try:
                    validation_sequence = self.data_queue.get(timeout=10)
                except:
                    continue

                if validation_sequence == -1:
                    #All the sequence from validation have been processed exiting
                    break
                
                if self.prev_scene_id == -1:
                    self.prev_scene_id = "scene0"#validation_sequence[0]["scene_id"]
                    self.curr_starting_frame = self.current_frame_id

            log.debug(f"tracker {self.tracker_type} validation sequence lenght {len(validation_sequence)}")
            #check if there is a scene change in the next list
            # index_change = min([i for i, el in enumerate(validation_sequence) if el["scene_id"] != self.prev_scene_id], default=-1)

            if index_change != -1:
                log.debug("New scene detected")
                validation_sequence_new_scene = validation_sequence[index_change:]
                validation_sequence = validation_sequence[:index_change]

            new_tracklets = self.generate_tracklets(validation_sequence)
            self.tracklet_list.append(new_tracklets)

            self.current_frame_id = self.current_frame_id + len(validation_sequence)
            log.debug("Tracklet generated")

            if index_change != -1:
                log.debug("combining tracklets")
                #the scene has change we combine tracklet, compute metrics, and reset the tracker
                self.pred_tracks = self.combine_tracklets()
                self.all_tracks.append(self.pred_tracks)
                self.all_scene_ids.append(self.prev_scene_id)
                self.all_seq_starting_frame.append(self.curr_starting_frame)

                #reseting tracker
                self.reset()

                #setting up for next iteration and next scene 
                self.prev_scene_id = validation_sequence_new_scene[0]["scene_id"]
                self.curr_starting_frame = self.current_frame_id
                validation_sequence = validation_sequence_new_scene
                continue

        self.data_queue.close()
        self.pred_tracks = self.combine_tracklets()
        self.all_tracks.append(self.pred_tracks)
        self.all_scene_ids.append(self.prev_scene_id)
        self.all_seq_starting_frame.append(self.curr_starting_frame)


    # def compute_tracking_metric(self):
    #     groundtruth_tracks_df = make_dataframe_from_tracks(self.groundtruth_tracks.values())
    #     pred_tracks_df = make_dataframe_from_tracks(self.pred_tracks)

    #     nb_gt = get_nb_det_per_frame_from_tracks(self.groundtruth_tracks.values()).values()
        
    #     return compute_mot_metric(groundtruth_tracks_df, pred_tracks_df, self.metric_threshold, nb_gt)

    # def generate_visualization(self):
    #     if self.conf["data_conf"]["image_plane"]:
    #         pred_scale = 8
    #     else:
    #         pred_scale = 4

    #     # log.debug(f"tracker {self.tracker_type} visu img {self.groundplane_images[0].shape}")
    #     # log.debug(f"tracker {self.tracker_type} visu roi {self.rois[0].shape}")

    #     visualization_result = make_tracking_vis(self.groundplane_images, flatten(self.all_tracks), f"{self.tracker_type} Tracking", self.rois, pred_scale, gt_points=self.gt_points)
    #     save_visualization_as_video(self.conf["training"]["ROOT_PATH"], {f"{self.tracker_type}":visualization_result}, self.conf["main"]["name"], self.epoch, out_type="mp4")

    # def generate_mot_result_file(self):
    #     for (scene_id, starting_frame, homography, tracks_list) in zip(self.all_scene_ids, self.all_seq_starting_frame, self.all_homography, self.all_tracks):
    #         mot_dict = generate_mot_list_from_tracks(tracks_list, homography,  self.conf["data_conf"]["homography_input_size"], self.conf["data_conf"]["homography_output_size"], self.conf["data_conf"]["hm_size"],  starting_frame)
    #         file_name = Path(self.conf["training"]["ROOT_PATH"] + f"/mot_results/MOT20-train/mot_{self.conf['main']['name']}_epoch_{self.epoch}_{self.tracker_type}/ground_{scene_id}.pth")
    #         file_name.parents[0].mkdir(parents=True, exist_ok=True)
    #         torch.save(mot_dict, str(file_name))


def cosine_similarity(vec1, vec2):
    return ((np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))) + 1) / 2

def flow_similarity(d1_pos, d1_flow, d2_pos, d2_flow):
        
    flow_vec = get_dir_vec_from_flow(d1_flow)
    
    if np.linalg.norm(d2_pos-d1_pos) == 0:
        return 1

    if flow_vec == [0,0]:
        return 1
    
    cos_sim = cosine_similarity(flow_vec, d2_pos-d1_pos)

    return cos_sim



def flow_similarity_dist(d1_pos, d1_flow, d1_index, d2_pos, d2_index):
        
    flow_vec = get_dir_vec_from_flow(d1_flow)
    
    predict_pos = d1_pos + np.array(flow_vec)*(d2_index - d1_index)

    dist = wf.euclidean(predict_pos, d2_pos)

    return dist

def appearance_similarity(reid1, reid2):
    return wf.euclidean(reid1, reid2)

# def weight_distance_tracklets_2d_flow(t1, t2,
#                                     sigma_color_histogram=0.3, sigma_motion=50, sigma_jump=0, sigma_distance=2, alpha=0.7,
#                                     cutoff_motion=0.1, cutoff_appearance=0.1,
#                                     max_distance=None,
#                                     use_color_histogram=True, debug=False, use_appearance=False):

#     def log(dev=-1, chs=-1, wm=-1, wa=-1, msg=''):
#         if debug:
#             print("head:{}:{} tail:{}:{} |dev:{:0.3f} w:{:0.3f}|color:{:0.3f} w:{:0.3f}| {}".format(t1.head.index, tuple(t1.head.position), 
#                                                                                                     t2.tail.index, tuple(t2.tail.position),
#                                                                                                     dev,wm, chs,wa, msg))
#     if max_distance is not None:
#         dist = wf.euclidean(t1.head.position, t2.tail.position)
#         if dist>max_distance:
#             return None             

#     # if an object is leaving the scene while another is entering the scene
#     # it is possible that the two trajectories will be connected. 
#     # To prevent this we simply avoid creating edges/links on the access points.
#     if t2.tail.access_point:
#         return None

#     # motion model
#     deviation = df.linear_motion(t1.head.indexes, t1.head.positions, 
#                                     t2.tail.indexes, t2.tail.positions)    
#     wm = np.exp(-deviation**2/sigma_motion**2)
#     if wm<cutoff_motion:
#         log(dev=deviation, wm=wm, msg='discarded: cutoff motion')
#         return None    
#     if use_appearance:
#         aps = 1-appearance_similarity(t1.head.color_histogram, t2.tail.color_histogram)
#         wap = np.exp(-aps**2/sigma_color_histogram**2)

#         return -((1-alpha)*wm + alpha*wap)  
#     else:
#         return -wm

def weight_distance_tracklets_2d_flow(t1, t2,
                                 sigma_color_histogram=0.3, sigma_motion=50, sigma_jump=0, sigma_distance=2, alpha=0.7,
                                 cutoff_motion=0.1, cutoff_appearance=0.1,
                                 max_distance=None,
                                 use_color_histogram=True, debug=False, use_appearance=False):
    
    def log(dev=-1, chs=-1, wm=-1, wa=-1, msg=''):
        if debug:
            print("head:{}:{} tail:{}:{} |dev:{:0.3f} w:{:0.3f}|color:{:0.3f} w:{:0.3f}| {}".format(t1.head.index, tuple(t1.head.position), 
                                                                                                    t2.tail.index, tuple(t2.tail.position),
                                                                                                    dev,wm, chs,wa, msg))
    weights = []
    
    dist = wf.euclidean(t1.head.position, t2.tail.position)
    if dist>max_distance:
        return None      

    weights.append( np.exp(-dist**2/sigma_distance**2))

    if use_appearance:
        aps = 1-appearance_similarity(t1.tail.color_histogram, t2.head.color_histogram)
        weights.append( np.exp(-aps**2/sigma_color_histogram**2) ) 

    # if an object is leaving the scene while another is entering the scene
    # it is possible that the two trajectories will be connected. 
    # To prevent this we simply avoid creating edges/links on the access points.
    
    if t2.tail.access_point:
        return None
    
    # # motion model
    # deviation = df.linear_motion(t1.head.indexes, t1.head.positions, 
    #                              t2.tail.indexes, t2.tail.positions)    
    # wm = np.exp(-deviation**2/sigma_motion**2)

    jump = t1.diff_index(t2)

    return -np.exp(-(jump-1)*sigma_jump) * np.prod(weights) #-wm

def weight_distance_detections_2d_with_flow(d1, d2,
                                  sigma_jump=1, sigma_distance=2,
                                  sigma_color_histogram=0.3, sigma_box_size=0.3,  sigma_flow=3,
                                  max_distance=20,
                                  use_color_histogram=True, use_bbox=True, use_appearance=True):
    weights = []

    dist = wf.euclidean(d1.position, d2.position)
    if dist>max_distance:
        return None
    weights.append( np.exp(-dist**2/sigma_distance**2) )    
    
    if use_appearance:
        aps = 1-appearance_similarity(d1.color_histogram, d2.color_histogram)
        weights.append( np.exp(-aps**2/sigma_color_histogram**2) )  

    # if use_color_histogram:
    #     chs = 1-df.color_histogram_similarity(d1.color_histogram, d2.color_histogram)
    #     weights.append( np.exp(-chs**2/sigma_color_histogram**2) )
    
    if use_bbox:
        bss = 1-df.bbox_size_similarity(d1.bbox, d2.bbox)
        weights.append( np.exp(-bss**2/sigma_box_size**2) )
    
    # if use_flow:
    #     fls = flow_similarity_dist(d1.position, d1.flow, d1.index, d2.position, d2.index)
    #     # fls = 1-flow_similarity(d1.position, d1.flow, d2.position, d2.flow)
    #     weights.append(np.exp(-fls**2/sigma_flow**2) )
    
    jump = d1.diff_index(d2)
    # log.debug(-np.exp(-(jump-1)*sigma_jump) * np.prod(weights))
    # if (-np.exp(-(jump-1)*sigma_jump) * np.prod(weights)) > -0.1:
    #     log.debug(-np.exp(-(jump-1)*sigma_jump) * np.prod(weights))
    #     return -0.1
    # if (-np.exp(-(jump-1)*sigma_jump) * np.prod(weights)) < -0.9:
    #     return -0.9

    return -np.exp(-(jump-1)*sigma_jump) * np.prod(weights)
    #return -np.exp(-(jump-1)*a) * np.exp(-distance**2/sigma_distance**2)

# class Detection2DAppearance(mot3d.Detection2D):
#     def __init__(self, index, position=None, confidence=0.5, id=None, 
#                     view=None, color_histogram=None, bbox=None, appearance=None):    
#         super().__init__(index, position, confidence, id, view, color_histogram, bbox)
#         self.appearance = appearance

class MuSSPTracker(BaseTracker):
    
    def __init__(self, conf, data_queue, use_appearance, *args, **kwargs):
        super().__init__(conf, data_queue, *args, **kwargs)
        self.use_appearance = use_appearance

        if use_appearance:
            self.tracker_type = "mussp_reid"
        else:
            self.tracker_type = "mussp"

        self.dummy_color_histogram = [np.ones((128,))]*3
        self.dummy_bbox = [0,0,1,1]

        conf["sigma_jump"] = 1
        conf["sigma_distance"] = 10#3 
        conf["sigma_flow"] = 8#10

        self.weight_distance = lambda d1, d2: weight_distance_detections_2d_with_flow(d1, d2,
                                                                        sigma_jump=conf["sigma_jump"], sigma_distance=conf["sigma_distance"],
                                                                        sigma_color_histogram=0.3, sigma_box_size=0.3, sigma_flow=conf["sigma_flow"], #3.5 # 5
                                                                        max_distance=50,
                                                                        use_color_histogram=False, use_bbox=False, use_appearance=self.use_appearance)

        self.weight_confidence = lambda d: wf.weight_confidence_detections_2d(d, mul=1, bias=0)

        # self.weight_distance_t = lambda t1, t2: wf.weight_distance_tracklets_2d(t1, t2, max_distance=None,
        #                                                                 sigma_color_histogram=0.3, sigma_motion=50, alpha=0.7,
        #                                                                 cutoff_motion=.2, cutoff_appearance=0.1,
        #                                                                 use_color_histogram=False)

        self.weight_distance_t = lambda t1, t2: weight_distance_tracklets_2d_flow(t1, t2, max_distance=50,
                                                                        sigma_color_histogram=50, sigma_motion=5, sigma_jump=conf["sigma_jump"], sigma_distance=conf["sigma_distance"], alpha=0.7,
                                                                        cutoff_motion=.2, cutoff_appearance=0.1,
                                                                        use_color_histogram=False, use_appearance=self.use_appearance)


        
        self.weight_confidence_t = lambda t: wf.weight_confidence_tracklets_2d(t, mul=1, bias=0)

    def reset(self):
        BaseTracker.reset(self)

    def generate_tracklets(self, step_dict_list):
        
        if len(step_dict_list) == 0:
            return list()

        detections = list()
        for i, frame_dict in enumerate(step_dict_list):
            curr_detections, appearance_features = frame_dict
            # curr_detections = frame_dict[f"{self.det_type}_{self.time_id}{self.motion_dir}_points"] #frame_dict[f"gt_points_{self.time_id}"] #  #np.append(frame_dict["processed_results"]["pred_point_flow"], np.array([[51,51]]), 0) # + [[51,51]]
            # curr_flow = frame_dict[f"offset_{self.time_id}_{self.next_time_id}{self.motion_dir}"]
            # curr_flow = (curr_flow / torch.clamp(torch.max(curr_flow, 0, keepdim=True)[0], min=0.05, max=1.0))
            # log.debug(curr_detections)
            # detections.extend([Detection2DFlow(self.current_frame_id + i, det, color_histogram=self.dummy_color_histogram, bbox=self.dummy_bbox, flow=curr_flow[:,det[1],det[0]]) for det in curr_detections])

            for j, det in enumerate(curr_detections):

                if self.use_appearance:
                    appearance = appearance_features[j]
                else:
                    appearance = None
                
                det2dflow = mot3d.Detection2D(self.current_frame_id + i, det, color_histogram=appearance, bbox=self.dummy_bbox)
                detections.append(det2dflow)

        # with stdout_redirected():
        # with suppress_stdout():
        g = mot3d.build_graph(detections, weight_source_sink=0.1,
                            max_jump=4, verbose=False,
                            weight_confidence=self.weight_confidence,
                            weight_distance=self.weight_distance)

        if g is None:
            #the graph is empty, we return an empty tracklet dict
            return list()

        # with suppress_stdout():
        tracklets = mot3d.solve_graph(g, verbose=False, method='muSSP')

        return tracklets

    def combine_tracklets(self):
        # log.debug("Combining")
        
        # log.debug(len(self.tracklet_list))
        tracklets = flatten(self.tracklet_list)

        tracklets_cut = []
        for traj in tracklets:
            tracklets_cut += mot3d.split_trajectory_modulo(traj, length=5)

        tracklets_cut = mot3d.remove_short_trajectories(tracklets_cut, th_length=2)
        # log.debug(len(tracklets))

        # for tracklet in tracklets:
        #     # print(tracklet)
        #     print("-------")
        #     print( " -- ".join([f"({det.index}, {str(det.position)}, {str(det.flow)})" for det in tracklet]))
        #split longer tracklet in short one not needed in when generating tracklet in batch?
        # for traj in trajectories:
        #     tracklets += mot3d.split_trajectory_modulo(traj, length=5)
        # tracklets = mot3d.remove_short_trajectories(tracklets, th_length=2)
        # log.debug(tracklets[0])
        detections_tracklets = [mot3d.DetectionTracklet2D(tracklet) for tracklet in tracklets_cut]
        
        # with stdout_redirected():
        # with suppress_stdout():
        g = mot3d.build_graph(detections_tracklets, weight_source_sink=0.1,
                                max_jump=20, verbose=False,
                                weight_confidence=self.weight_confidence_t,
                                weight_distance=self.weight_distance_t)    
        if g is None:
            log.warning(f"tracker {self.tracker_type} Combining tracklet there is not a single path between sink and source nodes! return empty tracking")
            return list() 
        
        # with suppress_stdout():
        trajectories = mot3d.solve_graph(g, verbose=False, method='muSSP')

        tracks = dict()



        for i, track in enumerate(trajectories):
            tracks[i] = Track(i)
            
            tracklet = trajectory.concat_tracklets([x.tracklet for x in track])
            tracklet = trajectory.interpolate_trajectory(tracklet)
            # tracklet = tracklet.tracklet
            for det in tracklet:
                tracks[i].add_detection(det.index, det.position[0], det.position[1])
        
        # log.debug(len(tracks.values()))
        # for v in tracks.values():
        #     print(v)
        # log.debug("combining done")
        return list(tracks.values())