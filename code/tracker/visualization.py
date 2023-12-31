import colorsys

import cv2
import numpy as np

# adapted from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py
def generate_colors(N):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*c)), hsv))
    perm = np.random.permutation(N)
    colors = [colors[idx] for idx in perm]
    
    return colors

def make_tracking_vis(frames_ground, start_ind, tracks, frame_id, start_ind_track, ROI_mask):
    out_frames = list()
    
    colors = generate_colors(len(tracks))
    
    for i, frame in enumerate(frames_ground):
        index_in_track = i + start_ind - start_ind_track

        people_loc = []
        for p_id, (track, track_frame_id) in enumerate(zip(tracks, frame_id)):
            if  len(in_track_id := np.argwhere(index_in_track == track_frame_id)) != 0:
                people_loc.append((p_id, track[in_track_id[0][0]]))

        vis = make_frame_visu_track(frame, people_loc, colors, roi=ROI_mask)
        
        out_frames.append(vis)
        
    return out_frames

def make_tracking_vis_old(frames, tracks_list, title, rois, pred_scale, gt_points=None):
    out_frames = list()
    
    colors = generate_colors(len(tracks_list))

    for i, frame in enumerate(frames):
        
        # frame_inv = inverse_img_norm(frame)
        people_loc = []
        for track in tracks_list:
            if i in track.detections:
                people_loc.append((track.person_id, track.detections[i]))

        if gt_points is not None:
            gt_points_curr = gt_points[i]
        else:
            gt_points_curr = None

        vis = make_frame_visu_track(frame, people_loc, colors, title=title, roi=rois[i], pred_scale=pred_scale, gt_points=gt_points_curr)
        
        out_frames.append(vis)
        
    return out_frames
        


def make_frame_visu_track(input_img, partial_tracks, colors, title=None, roi=None, pred_scale=1, gt_points=None):
    
    # out_image = np.ascontiguousarray(np.array(input_img * 255, dtype = np.uint8))
    
    # if roi is not None:
    #     roi = np.array(roi.squeeze() * 255, dtype = np.uint8)
    #     roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE )
    #     if pred_scale != 1:
    #         roi = cv2.resize(roi, tuple([c*pred_scale for c in roi.shape[:2][::-1]]), interpolation=cv2.INTER_NEAREST)

    #     if out_image.shape[:2] != roi.shape[:2]:
    #         out_image = cv2.resize(out_image, roi.shape[:2])

    #     out_image = cv2.addWeighted(out_image, 0.8, roi, 0.2, 0)
    input_img = np.array(input_img, dtype=np.uint8)

    # Create a copy of the input image
    out_image = input_img.copy()
    
    # Visualize the ROI if provided
    if roi is not None:
        roi = np.array(roi.squeeze() * 255, dtype=np.uint8)
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_BONE)
        out_image = cv2.addWeighted(out_image, 0.8, roi, 0.4, 0)
    
    for track_id, partial_track in partial_tracks:
        # print(track_id)
        partial_track = np.array(partial_track, np.int32)*pred_scale
        # print(partial_track)
        # print(partial_track.shape)
        # print(colors)
        
        if partial_track.shape[0] != 0:
            cv2.drawMarker(out_image, (partial_track[0], partial_track[1]), colors[track_id % len(colors)], cv2.MARKER_STAR, markerSize=6, thickness=2) #MarkerType[, markerSize[, thickness[, line_type]]]]	)
            cv2.putText(out_image, str(track_id), (partial_track[0] + 8, partial_track[1]+3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[track_id % len(colors)], thickness=1)
            # cv2.polylines(out_image, [partial_track.reshape((-1, 1, 2))], False, colors[track_id], thickness=2)

#             ax.plot(partial_track[:,0], partial_track[:,1], linewidth=5, c=colormap((track_id*10) % nb_track))
        if gt_points is not None:
            for point in gt_points:
                out_image = cv2.drawMarker(out_image, tuple(point*pred_scale), [0,190,0], cv2.MARKER_SQUARE, markerSize=4, thickness=1) #MarkerType[, markerSize[, thickness[, line_type]]]]	)  


    # out_image = cv2.copyMakeBorder(out_image, 32, 32, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
    
    # #Title Bar
    # if title is not None:
    #     textSize, baseline = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 1)
    #     out_image = cv2.putText(out_image, title, (int(out_image.shape[1] / 2 - textSize[0] / 2), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.8, [255,255,255], thickness=1)
    
#     #Bottom text add gt and metric if availble
#     if count is not None and count_det is not None:
#         bottom = f"Count - Hm {count} - Det {count_det} "
#     else:
#         bottom = ""
        
#     if gt_points is not None:
#         bottom = bottom + f"Gt - {str(gt_points.shape[0])} "

#     if prec is not None:
#         bottom = bottom + f"|| Prec - {prec} Rec - {rec}"
    
#     textSize, baseline = cv2.getTextSize(bottom, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#     out_image = cv2.putText(out_image, bottom, (int(out_image.shape[1] / 2 - textSize[0] / 2), 561), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255,255,255], thickness=1)
    
    return out_image