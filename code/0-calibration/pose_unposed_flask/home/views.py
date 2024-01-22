from flask import Blueprint, render_template, redirect, url_for, current_app
import os
import shutil

home = Blueprint('home', __name__)

@home.route('/')
def homepage():
    global reconstruction_uploaded, matches_uploaded, images_uploaded, features_uploaded, calibrated

    # check which files already exist
    reconstruction_uploaded = os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'reconstruction.json'))

    matches_uploaded = os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'matches'))

    images_uploaded = os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'images'))

    features_uploaded = os.path.exists(os.path.join(current_app.config['UPLOAD_FOLDER'], 'features'))

    ready_for_calibration = features_uploaded and images_uploaded and matches_uploaded and reconstruction_uploaded

    return render_template('home.html', 
                            reconstruction_uploaded=reconstruction_uploaded,
                            matches_uploaded=matches_uploaded,
                            images_uploaded=images_uploaded,
                            features_uploaded=features_uploaded,
                            ready_for_calibration=ready_for_calibration)


@home.route('/opensfm/segmentation')
def segmentation():
    return render_template('opensfm_segmentation.html')


@home.route('/delete', methods=['POST'])
def delete_uploads():
    shutil.rmtree(current_app.config['UPLOAD_FOLDER'])
    os.makedirs(current_app.config['UPLOAD_FOLDER'])
    return redirect(url_for('home'))