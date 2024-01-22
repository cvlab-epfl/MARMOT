from flask import Flask, request, send_from_directory, render_template
import os
# Load Cameras

import os
import sys
import numpy as np
import json
from home.views import home
from calibration.views import uploads
from home.api import api


# %matplotlib widget

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = 'uploads'

# load the config file
app.config['CONFIG'] = None

app.register_blueprint(home)
app.register_blueprint(uploads)
app.register_blueprint(api)


from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os


reconstruction_uploaded = False
matches_uploaded = False
images_uploaded = False
features_uploaded = False
calibrated = False


from flask import request, flash
import os
from flask import request, redirect, render_template
from werkzeug.utils import secure_filename
import os
import json


@app.route('/calibrate', methods=['GET', 'POST'])
def calibrate():
    # select camera to calibrate


    return render_template('calibrate.html')





import cv2
from flask import send_file
import io

import pickle
import gzip







# @app.route()
# def multiview_calibration():
#     """
#     Perform Multiview Calibration using multiview_calib
#     """
#     pass

def choose_images_for_visualisation():
    """
    Choose images to visualise the calibration through multiview_calib
    """
    filenames_path = os.path.join(app.config['UPLOAD_FOLDER'], 'filenames.json')
    filenames = {}
    # TODO:
    # Insert user interface magic here

    # Create filenames.json file
    with open(filenames_path, 'w') as f:
        json.dump(filenames, f, indent=4)

    pass


    





if __name__ == '__main__':


    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # make ba config file
    ba_config_path = os.path.join(app.config['UPLOAD_FOLDER'], 'ba_config.json')
    ba_config = {
        "each_training": 1,
        "each_visualisation": 1,
        "th_outliers_early": 1000.0,
        "th_outliers": 50,
        "optimize_points": True,
        "optimize_camera_params": True,
        "bounds": True,  
        "bounds_cp": [ 
            0.3, 0.3, 0.3,
            2, 2, 2,
            10, 10, 10, 10,
            0.01, 0.01, 0, 0, 0
        ],
        "bounds_pt": [
            1000,
            1000,
            1000
        ],
        "max_nfev": 200,
        "max_nfev2": 200,
        "ftol": 1e-08,
        "xtol": 1e-08,  
        "loss": "linear",
        "f_scale": 1,
        "output_path": "output/bundle_adjustment/",
        }
    
    with open(ba_config_path, 'w') as f:
        json.dump(ba_config, f, indent=4)
    
    app.run(debug=True)
