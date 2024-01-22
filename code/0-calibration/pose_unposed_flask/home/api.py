from flask import Blueprint, request, redirect, send_file, jsonify, current_app
import os
import shutil
import pickle
import gzip
import io
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from calibration.defs import load_features_and_matches, denormalized_image_coordinates, draw_features

import json

api = Blueprint('api', __name__)


@api.route('/api/cameras', methods=['GET'])
def get_cameras():
    # This is a placeholder. Replace with your actual logic.
    with open(os.path.join(current_app.config['UPLOAD_FOLDER'], 'reconstruction.json'), 'r') as f:
        reconstruction_data = json.load(f)[0]

    
    cameras = set([key for key in reconstruction_data['cameras'].keys()])
    return jsonify(list(cameras))


@api.route('/camera-data', methods=['POST'])
def handle_camera_data():
    config = api.config.get('CONFIG')
    data = request.get_json()
    for camera_name, file_paths in data.items():
        # Handle the file paths for this camera
        print(f"Camera Name: {camera_name}")
        for file_path in file_paths:
            # symlink the file to data folder
            print(f"File Path: {file_path}")
            if config is not None:
                if config.get('data') is not None:
                    camera_dir = os.path.join(config.get('data'), camera_name)
                    os.mkdir(camera_dir)
                    os.symlink(file_path, os.path.join(camera_dir, file_path.split('/')[-1]))
    return jsonify(footage_dict)

@api.route('/calibration-data', methods=['POST'])
def handle_calibration_data():
    data = request.get_json()
    for camera_name, file_paths in data.items():
        # Handle the file paths for this camera
        print(f"Camera Name: {camera_name}")
        for file_path in file_paths:
            print(f"File Path: {file_path}")
    return jsonify(calib_footage_dict)


@api.route('/api/camera/matches/<int:camera_id>/<int:frame_id>/<int:camera_id_2>/<int:frame_id_2>/', methods=['GET'])
def get_camera_matches(camera_id, frame_id, camera_id_2, frame_id_2):
    images_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'images')
    image_names = [name for name in os.listdir(images_dir) if f"cam{camera_id}_" in name]
    image_names.sort()

    image_names_2 = [name for name in os.listdir(images_dir) if f"cam{camera_id_2}_" in name]
    image_names_2.sort()

    matches_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'matches')

    with gzip.open(os.path.join(matches_path, image_names[frame_id % len(image_names)] + '_matches.pkl.gz'), 'rb') as f:
        matches = pickle.load(f)

    
    match_img = matches[image_names_2[frame_id_2 % len(image_names_2)]] if image_names_2[frame_id_2 % len(image_names_2)] in matches else None

    if match_img is None:
        return None

    return match_img if len(match_img) > 0 else None

@api.route('/api/click', methods=['POST'])
def handle_click():
    data = request.get_json()
    x = data['x']
    y = data['y']

    print(x, y)

    # Do something with x and y...

    return '', 204

@api.route('/api/camera/<int:camera_id>/frame/<int:frame_id>/camera/<int:camera_id_2>/frame/<int:frame_id_2>/', methods=['GET'])
def get_camera_frame(camera_id, frame_id, camera_id_2, frame_id_2):
    displayOption = request.args.get('displayOption', 'Display features')

    def get_image(cam, frame):
        images_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'images')
        image_names = [name for name in os.listdir(images_dir) if f"cam{cam}_" in name]
        image_names.sort()
        img_path = os.path.join(images_dir, image_names[frame % len(image_names)])
        img = cv2.imread(img_path)
        features, matches = load_features_and_matches(img_path)
        
        return img, features


    img1, features1 = get_image(camera_id, frame_id)
    img2, features2 = get_image(camera_id_2, frame_id_2)
    
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    features1 = denormalized_image_coordinates(features1['points'], img1.shape[1], img1.shape[0])
    features2 = denormalized_image_coordinates(features2['points'], img2.shape[1], img2.shape[0])

    if displayOption == 'Display features':
        img1 = draw_features(img1, features1)
        img2 = draw_features(img2, features2)

        img_stacked = np.hstack((img1, img2))
    
    if displayOption == 'Display matches':
        matches = get_camera_matches(camera_id, frame_id, camera_id_2, frame_id_2)
        

        if matches is not None:
            img1 = draw_features(img1, features1[matches[:,0]], colour=(0, 255, 0))
            img2 = draw_features(img2, features2[matches[:,1]], colour=(0, 255, 0))
            img_stacked = np.hstack((img1, img2))

            for match in matches:
                print(match[0])
                cv2.line(img_stacked, 
                        (int(features1[match[0],0]), int(features1[match[0],1])), 
                        (int(features2[match[1],0] + img1.shape[1]), int(features2[match[1],1])), 
                        (0, 255, 0), 10)
        else:
            img_stacked = np.hstack((img1, img2))

    _, img_encoded = cv2.imencode('.jpg', img_stacked)
    img_io = io.BytesIO(img_encoded.tobytes())
    return send_file(img_io, mimetype='image/jpeg')