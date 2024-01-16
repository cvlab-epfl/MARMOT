from flask import Flask, request, send_from_directory, render_template
import os
# Load Cameras

import os
import sys
import numpy as np
import json

# sys.path.append('..')
# from configs.arguments import get_config_dict
# from utils.multiview_utils import Camera, Calibration, MultiviewVids
# from utils.io_utils import write_json, load_json
# from utils.metadata_utils import get_cam_names
# from utils.coordinate_utils import update_reconstruction, point_in_polygon, project_to_ground_plane_cv2
# from utils.plot_utils import rotation_matrix, perp

# from scipy.spatial.transform import Rotation as R
# from skspatial.objects import Point, Vector, Plane, Points, Line
# import pyransac3d as pyrsc

# import copy
# import cv2
# import ipywidgets as widgets
# import ipympl
# import matplotlib.pyplot as plt


# %matplotlib widget

app = Flask(__name__)

from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'uploads'


reconstruction_uploaded = False
matches_uploaded = False
images_uploaded = False
features_uploaded = False
calibrated = False


def allowed_file(filename):
    # return '.' in filename and \
    #        filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']
    return True

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return 'No file part', 400
    files = request.files.getlist('files[]')
    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filenames.append(filename)
    return 'Files uploaded successfully', 200


    

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory('/path/to/save/uploads', filename)


def return_unposed(reconstruction_data):
    if reconstruction_data is None:
        print("Reconstruction not found. Please run 0-calibration/2-extrinsics.py first.")
        return None
    
    list_of_cams = set([key for key in reconstruction_data['cameras'].keys()])
    print(list_of_cams)
    for shot in reconstruction_data['shots'].values():
        print(shot)
        if shot['camera'] in list_of_cams:
            list_of_cams.remove(shot['camera'])
    
    return list(list_of_cams)




# arv_copy = sys.argv
# sys.argv = ['pop']
# sys.argv.append('-cfg')
# config_path = os.path.abspath('../../project_config.yaml')
# sys.argv.append(config_path)
# sys.argv.append('-dr')
# root_path = os.path.abspath('../../data/')
# sys.argv.append(root_path)
# sys.argv.append('-l')
# sys.argv.append('info')

# # load arguments from the arg parser
# config = get_config_dict()
# data_root = config["main"]["data_root"]
# calib_dir = os.path.join(data_root, '0-calibration', 'calibs')
# video_dir = os.path.join(data_root, 'raw_data', 'footage')
# reconstruction_dir = os.path.join(data_root, '0-calibration', 'opensfm', 'undistorted', 'reconstruction.json')
# omni_tag = '360'


# reconstruction_dir = os.path.join(data_root, '0-calibration', 'opensfm', 'reconstruction.json')
# if not os.path.exists(reconstruction_dir):
#     print( "Reconstruction not found. Please run 0-calibration/2-extrinsics.py first.")
    

# def load_libs():



@app.route('/')
def home():
    global reconstruction_uploaded, matches_uploaded, images_uploaded, features_uploaded, calibrated

    # check which files already exist
    reconstruction_uploaded = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'reconstruction.json'))

    matches_uploaded = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'matches'))

    images_uploaded = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'images'))

    features_uploaded = os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'features'))

    ready_for_calibration = features_uploaded and images_uploaded and matches_uploaded and reconstruction_uploaded

    return render_template('home.html', 
                            reconstruction_uploaded=reconstruction_uploaded,
                            matches_uploaded=matches_uploaded,
                            images_uploaded=images_uploaded,
                            features_uploaded=features_uploaded,
                            ready_for_calibration=ready_for_calibration)

from flask import request, flash
from werkzeug.utils import secure_filename
import os

@app.route('/upload/reconstruction', methods=['GET', 'POST'])
def upload_reconstruction():
    global reconstruction_uploaded
    

    if request.method == 'POST':
        if 'file' not in request.files:  # Changed from request.file to request.files
            print('No file uploaded')
            return redirect(request.url)
        file = request.files['file']  # Changed from request.file to request.files
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('json'):
            print('File uploaded successfully')
            reconstruction_uploaded = True
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(request.url)#url_for('uploaded_file', filename=filename))
        
    # Check if reconstruction.json exists
    reconstruction_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reconstruction.json')
    if os.path.exists(reconstruction_path):
        with open(reconstruction_path, 'r') as f:
            reconstruction_data = json.load(f)
    else:
        reconstruction_data = None
    unposed = return_unposed(reconstruction_data[0])
    print(unposed)
    return render_template('upload_reconstruction.html', reconstruction_data=reconstruction_data[0], unposed=unposed)

@app.route('/upload/matches', methods=['GET', 'POST'])
def upload_matches():
    global matches_uploaded
    if request.method == 'POST':
        if 'files[]' not in request.files: 
            print('No matches uploaded')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        print('Uploaded {} matches'.format(len(files)))
        matches_uploaded = True
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'matches')):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'matches'))
        for file in files:
            if file.filename == '' or not file.filename.endswith('pkl.gz'):
                continue
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'matches', filename))
        return redirect(request.url)
    
    # Check if matches exist
    matches_path = os.path.join(app.config['UPLOAD_FOLDER'], 'matches')
    if os.path.exists(matches_path):
        matches = os.listdir(matches_path)
    else:
        matches = None

    return render_template('upload_matches.html', matches=matches)

@app.route('/upload/images', methods=['GET', 'POST'])
def upload_images():
    global images_uploaded
    if request.method == 'POST':
        if 'files[]' not in request.files: 
            print('No images uploaded')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        print('Uploaded {} images'.format(len(files)))
        images_uploaded = True
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'images')):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'images'))
        for file in files:
            if file.filename == '' or not file.filename.endswith('jpg', 'jpeg', 'png'):
                continue
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'images', filename))
        return redirect(request.url)
    
    # Check if images exist
    images_path = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
    if os.path.exists(images_path):
        images = os.listdir(images_path)
    else:
        images = None

    return render_template('upload_images.html', images=images)

@app.route('/upload/features', methods=['GET', 'POST'])
def upload_features():
    global features_uploaded
    if request.method == 'POST':
        if 'files[]' not in request.files: 
            print('No features uploaded')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        print('Uploaded {} features'.format(len(files)))
        features_uploaded = True
        if not os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], 'features')):
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'features'))
        for file in files:
            if file.filename == '' or not file.filename.endswith('npz'):
                continue
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'features', filename))
        return redirect(request.url)
    
    # Check if features exist
    features_path = os.path.join(app.config['UPLOAD_FOLDER'], 'features')
    if os.path.exists(features_path):
        features = os.listdir(features_path)
    else:
        features = None

    return render_template('upload_features.html', features=features)

@app.route('/calibrate', methods=['GET', 'POST'])
def calibrate():
    # select camera to calibrate


    return render_template('calibrate.html')

import shutil

@app.route('/delete', methods=['POST'])
def delete_uploads():
    shutil.rmtree(app.config['UPLOAD_FOLDER'])
    os.makedirs(app.config['UPLOAD_FOLDER'])
    return redirect(url_for('home'))

@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    # This is a placeholder. Replace with your actual logic.
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'reconstruction.json'), 'r') as f:
        reconstruction_data = json.load(f)[0]

    
    cameras = set([key for key in reconstruction_data['cameras'].keys()])
    return jsonify(list(cameras))

import cv2
from flask import send_file
import io


def draw_features(img, points, colour = (0, 0, 255)):
    if colour != (0, 0, 255):
        point_size = 5
    else:
        point_size = 2
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), point_size, colour, -1)
    return img

import pickle
import gzip

def load_features_and_matches(img_path):
    features_path = os.path.join(app.config['UPLOAD_FOLDER'], 'features')
    matches_path = os.path.join(app.config['UPLOAD_FOLDER'], 'matches')
    img_name = os.path.basename(img_path)
    features = np.load(os.path.join(features_path, img_name + '.features.npz'))

    with gzip.open(os.path.join(matches_path, img_name + '_matches.pkl.gz'), 'rb') as f:
        matches = pickle.load(f)

    return features, matches

def denormalized_image_coordinates(
        norm_coords: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        size = max(width, height)
        p = np.empty((len(norm_coords), 2))
        # handle the case where only a single point is passed
        if len(norm_coords.shape) == 1:
            norm_coords = norm_coords.reshape(1, -1)

        p[:, 0] = norm_coords[:, 0] * size - 0.5 + width / 2.0
        p[:, 1] = norm_coords[:, 1] * size - 0.5 + height / 2.0
        return p


@app.route('/api/camera/matches/<int:camera_id>/<int:frame_id>/<int:camera_id_2>/<int:frame_id_2>/', methods=['GET'])
def get_camera_matches(camera_id, frame_id, camera_id_2, frame_id_2):
    images_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
    image_names = [name for name in os.listdir(images_dir) if f"cam{camera_id}_" in name]
    image_names.sort()

    image_names_2 = [name for name in os.listdir(images_dir) if f"cam{camera_id_2}_" in name]
    image_names_2.sort()

    matches_path = os.path.join(app.config['UPLOAD_FOLDER'], 'matches')

    with gzip.open(os.path.join(matches_path, image_names[frame_id % len(image_names)] + '_matches.pkl.gz'), 'rb') as f:
        matches = pickle.load(f)

    
    match_img = matches[image_names_2[frame_id_2 % len(image_names_2)]] if image_names_2[frame_id_2 % len(image_names_2)] in matches else None

    if match_img is None:
        return None

    return match_img if len(match_img) > 0 else None
        

@app.route('/api/camera/<int:camera_id>/frame/<int:frame_id>/camera/<int:camera_id_2>/frame/<int:frame_id_2>/', methods=['GET'])
def get_camera_frame(camera_id, frame_id, camera_id_2, frame_id_2, highlight_matches=True):
    # This is a placeholder. Replace with your actual logic.
    images_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'images')
    image_names = [name for name in os.listdir(images_dir) if f"cam{camera_id}_" in name]
    image_names.sort()

    img = cv2.imread(os.path.join(images_dir, image_names[frame_id % len(image_names)]))

    features, matches = load_features_and_matches(image_names[frame_id % len(image_names)])

    image_names_2 = [name for name in os.listdir(images_dir) if f"cam{camera_id_2}_" in name]
    image_names_2.sort()
    features_2, _ = load_features_and_matches(image_names_2[frame_id_2 % len(image_names_2)])

    img_2 = cv2.imread(os.path.join(images_dir, image_names_2[frame_id_2 % len(image_names_2)]))

    # if image shapes are different, resize
    if img.shape[0] != img_2.shape[0] or img.shape[1] != img_2.shape[1]:
        img = cv2.resize(img, (img_2.shape[1], img_2.shape[0]))

    img = draw_features(img, denormalized_image_coordinates(features['points'], img.shape[1], img.shape[0]))

    

    features, matches = load_features_and_matches(image_names[frame_id % len(image_names)])

    

    img_2 = draw_features(img_2, denormalized_image_coordinates(features_2['points'], img_2.shape[1], img_2.shape[0]))

    img = draw_features(img, denormalized_image_coordinates(features['points'], img.shape[1], img.shape[0]))

    if highlight_matches: 
        matches = get_camera_matches(camera_id, frame_id, camera_id_2, frame_id_2)
        if matches is not None:
            # print(matches)
            img = draw_features(img, denormalized_image_coordinates(features['points'][matches[:,0]], img.shape[1], img.shape[0]), colour = (0, 255, 0))
            img_2 = draw_features(img_2, denormalized_image_coordinates(features_2['points'][matches[:,1]], img_2.shape[1], img_2.shape[0]), colour = (0, 255, 0))

    
    # stack images horizontally
    img_stacked = np.hstack((img, img_2))

    # denormed_features = denormalized_image_coordinates(features['points'], img.shape[1], img.shape[0])
    # denormed_features_2 = denormalized_image_coordinates(features_2['points'], img_2.shape[1], img_2.shape[0])

    # draw lines between corresponding points
    if highlight_matches:
        if matches is not None:
            print("drawing correspondence lines")
            for match in matches:
                cv2.line(img_stacked, 
                         (
                             int(denormalized_image_coordinates(features['points'][match[0]], img.shape[1], img.shape[0])[0,0]),
                             int(denormalized_image_coordinates(features['points'][match[0]], img.shape[1], img.shape[0])[0,1])
                             ), 
                         (
                             int(denormalized_image_coordinates(features_2['points'][match[1]], img_2.shape[1], img_2.shape[0])[0,0] + img.shape[1]), 
                             int(denormalized_image_coordinates(features_2['points'][match[1]], img_2.shape[1], img_2.shape[0])[0,1])
                             ), 
                         (0, 255, 0), 10)

    # Convert the image to JPEG format
    _, img_encoded = cv2.imencode('.jpg', img_stacked)

    # Create a BytesIO object and save the JPEG image data to it
    img_io = io.BytesIO(img_encoded.tobytes())

    # Send the image data as a file
    return send_file(img_io, mimetype='image/jpeg')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(debug=True)
