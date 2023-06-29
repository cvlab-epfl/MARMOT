#!/bin/bash

# set the base path
base_path="/root/data"
opensfm_path="/OpenSfM"

# data path
data_path="${base_path}/0-calibration/opensfm/"

# opensfm path
opensfm_viewer_path="${opensfm_path}/viewer"

python3 ${opensfm_viewer_path}/server.py -d ${data_path}
