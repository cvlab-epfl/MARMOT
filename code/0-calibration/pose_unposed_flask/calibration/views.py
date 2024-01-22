from flask import Blueprint, request, redirect, render_template, current_app
import os
from werkzeug.utils import secure_filename
from calibration.defs import return_unposed
import json

uploads = Blueprint('uploads', __name__)


@uploads.route('/refine/upload/reconstruction', methods=['GET', 'POST'])
def upload_reconstruction():
    print("uploading reconstruction")
    reconstruction_data = None
    unposed = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename.endswith('.json'):
            print('File uploaded successfully')
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
            return redirect(request.url)

        print('No valid file uploaded')
        return redirect(request.url)

    reconstruction_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'reconstruction.json')
    if os.path.exists(reconstruction_path):
        with open(reconstruction_path, 'r') as f:
            reconstruction_data = json.load(f)
        unposed = return_unposed(reconstruction_data[0])

    return render_template('upload_reconstruction.html', reconstruction_data=reconstruction_data[0] if reconstruction_data else None, unposed=unposed)

@uploads.route('/refine/upload/matches', methods=['GET', 'POST'])
def upload_matches():
    global matches_uploaded
    if request.method == 'POST':
        if 'files[]' not in request.files: 
            print('No matches uploaded')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        print('Uploaded {} matches'.format(len(files)))
        matches_uploaded = True
        if not os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'matches')):
            os.makedirs(os.path.join(current_app.config['UPLOAD_FOLDER'], 'matches'))
        for file in files:
            if file.filename == '' or not file.filename.endswith('pkl.gz'):
                continue
            filename = secure_filename(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], 'matches', filename))
        return redirect(request.url)
    
    # Check if matches exist
    matches_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'matches')
    if os.path.exists(matches_path):
        matches = os.listdir(matches_path)
    else:
        matches = None

    return render_template('upload_matches.html', matches=matches)



@uploads.route('/refine/upload/images', methods=['GET', 'POST'])
def upload_images():
    images = None

    if request.method == 'POST':
        files = request.files.getlist('files[]')
        if files:
            print(f'Uploaded {len(files)} images')
            images_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'images')
            os.makedirs(images_path, exist_ok=True)
            for file in files:
                if file.filename.endswith(('jpg', 'jpeg', 'png')):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(images_path, filename))
            return redirect(request.url)

        print('No valid images uploaded')
        return redirect(request.url)

    images_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'images')
    if os.path.exists(images_path):
        images = os.listdir(images_path)

    return render_template('upload_images.html', images=images)

@uploads.route('/refine/upload/features', methods=['GET', 'POST'])
def upload_features():
    features = None

    if request.method == 'POST':
        files = request.files.getlist('files[]')
        if files:
            print(f'Uploaded {len(files)} features')
            features_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'features')
            os.makedirs(features_path, exist_ok=True)
            for file in files:
                if file.filename.endswith('npz'):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(features_path, filename))
            return redirect(request.url)

        print('No valid features uploaded')
        return redirect(request.url)

    features_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'features')
    if os.path.exists(features_path):
        features = os.listdir(features_path)

    return render_template('upload_features.html', features=features)