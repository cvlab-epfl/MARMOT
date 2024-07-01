from flask import Flask, render_template

def setup_app():
    import os
    import sys
    import cv2
    import numpy as np
    import subprocess
    import pyexif
    from pathlib import Path

    sys.path.append('..')

    from utils.log_utils import log
    from utils.multiview_utils import Camera
    from utils.metadata_utils import get_cam_names
    from configs.arguments import get_config_dict
    from utils.io_utils import write_json

    arv_copy = sys.argv
    sys.argv = ['pop']
    sys.argv.append('-cfg')
    config_path = os.path.abspath('../../project_config.yaml')
    sys.argv.append(config_path)
    sys.argv.append('-dr')
    root_path = os.path.abspath('../../data/')
    sys.argv.append(root_path)
    sys.argv.append('-l')
    sys.argv.append('info')

    return config_path, root_path

def load_cameras():
    # Load cameras
    try:
        config = get_config_dict()
    except:
        log.warning("No config file found. Using default values.")
        config = {}
        
    data_root = Path(config.get('main', {}).get('data_root', '/root/data'))
    omni_tag = config.get('calibration', {}).get('omni_tag', '360')
    # data_root = Path(config.get('main', {}).get('data_root', '/Users/grosche/Documents/GitHub/CVLAB/MARMOT/data'))
    # set up paths
    env_footage = data_root / 'raw_data' / 'footage'
    opensfm_data =  data_root / '0-calibration' / 'opensfm'
    opensfm_images = opensfm_data / 'images'
    extrinsics_path = data_root / '0-calibration' / 'extrinsics'

    cams = get_cam_names(env_footage)
    camera_models_overrides_dict = {}
    log.info(f"Found {(cams)} in environment footage.")
    len_360 = 0
    sample_images = []
    for cam in cams:
        camera = Camera(cam, newest=False)
        if omni_tag in camera.name:
            omni_cam = camera
            len_360 = len(camera)
            omni_imgs = camera.extract(range(0, len_360, 30))
            continue
        sample_images.append(camera.extract([10])[0])





app = Flask(__name__)

@app.route('/')
def home():
    config_path, root_path = setup_app()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)