import os
import json
import mimetypes
import shutil
import re
import numpy as np
import json
import cv2
import pandas as pd
import os
import gzip
import pickle

mimetypes.init()

def is_media_file(fileName: str) -> bool:
    """
        Returns True if fileName is a media file
        (i.e. a video or image file), False otherwise.

        Arguments:
        ----------
        fileName(str): name of file
    """
    mimestart = mimetypes.guess_type(fileName)[0]

    if mimestart != None:
        mimestart = mimestart.split('/')[0]

        if mimestart in ['video', 'image']:
            return True
    
    return False


def load_json(filename: str) -> dict:

    """
        Attempts to read json file 'filename'.
        Throws an exception if unable to do so.

        Arguments:
        ----------
        filename(str): path to json file
    """

    try:
        with open(os.path.abspath(filename)) as f:    
            data = json.load(f)
        return data
    except:
        raise ValueError("Unable to read JSON {}".format(filename))


def write_json(filename: str, data: dict) -> None:

    """
        Attempts to write json file 'filename'.
        Throws an exception if unable to do so.

        Arguments:
        ----------
        filename(str): path to json file
        data(dict): data to be written to json file
    """

    try:
        directory = os.path.dirname(os.path.abspath(filename))
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.abspath(filename), 'w') as f:
            json.dump(data, f, indent=2)
    except:
        raise ValueError("Unable to write JSON {}".format(filename))


def rgb2gray(image):
    dtype = image.dtype
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    return gray.astype(dtype)

def rmdir(directory):
    directory = os.path.abspath(directory)
    if os.path.exists(directory): 
        shutil.rmtree(directory)  

def mkdir(directory):
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key=alphanum_key)

def find_files(file_or_folder, hint=None, recursive=False):
    # make sure to use ** in file_or_folder when using recusive
    # ie find_files("folder/**", "*.json", recursive=True)
    import os
    import glob
    if hint is not None:
        file_or_folder = os.path.join(file_or_folder, hint)
    filenames = [f for f in glob.glob(file_or_folder, recursive=recursive)]
    filenames = sort_nicely(filenames)    
    filename_files = []
    for filename in filenames:
        if os.path.isfile(filename):
            filename_files.append(filename)                 
    return filename_files
     
def find_images(file_or_folder, hint=None):  
    filenames = find_files(file_or_folder, hint)
    filename_images = []
    for filename in filenames:
        _, extension = os.path.splitext(filename)
        if extension.lower() in [".jpg",".jpeg",".bmp",".tiff",".png",".gif"]:
            filename_images.append(filename)                 
    return filename_images  


def get_landmarks(opensfm_dir):
    """
    Creates landmarks.json from OpenSfM tracks.csv file.

    Args:
        opensfm_dir (str): path to OpenSfM directory

    Returns:
        landmarks (dict): dict({
            'image_name': dict({
                'landmarks': list([x, y]),
                'ids': list(track_id)
        })
    """
    tracks_path = os.path.join(opensfm_dir, 'tracks.csv')
    tracks = pd.read_csv(tracks_path, delimiter='\t', skiprows=1, names=['image', 'track_id', 'feature_id', 'x', 'y', 'scale', 'r', 'g', 'b', 'segmentation', 'instance'])

    def process_group(group):
        img_path = os.path.join(opensfm_dir, 'images', group.name)
        img = cv2.imread(img_path)
        group['x'] = ((group['x'] + 0.5) * img.shape[1]).astype(int)
        group['y'] = ((group['y'] + 0.5) * img.shape[0]).astype(int)
        return {'ids': group['track_id'].values.tolist(), 'landmarks': group[['x', 'y']].values.tolist()}

    landmarks = tracks.groupby('image').apply(process_group).to_dict()

    return landmarks

def get_landmarks_global(opensfm_dir):
    """
    Creates landmarks_global.json from OpenSfM reconstruction.json file.

    Args:
        opensfm_dir (str): path to OpenSfM directory

    Returns:
        landmarks_global (dict): dict({
            'ids': list(track_id),
            'points': list([x, y, z])
        })
    """
    reconstruction_path = os.path.join(opensfm_dir, 'reconstruction.json')
    with open(reconstruction_path) as f:
        reconstruction = json.load(f)[0]

    landmarks_global = {}
    landmarks_global['ids'] = []
    landmarks_global['points'] = []
    for key, value in reconstruction['points'].items():
        landmarks_global['ids'].append(int(key))
        landmarks_global['points'].append(value['coordinates'])

    return landmarks_global

def get_intrinsics(opensfm_dir):
    """
    Concatenates the intrinsics of all cameras into a single dictionary.

    Args:
        opensfm_dir (str): path to OpenSfM directory (undistorted)

    Returns:
        intrinsics (dict): dict({
            'image_name': dict({
                'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                'distortion': list([k1, k2, p1, p2, k3])
        })
    """

    intrinsics = {}
    reconstruction_path = os.path.join(opensfm_dir, 'reconstruction.json')
    with open(reconstruction_path) as f:
        reconstruction = json.load(f)[0]


        # for camera in reconstruction['cameras']:
        #     intrinsics[camera['image']] = {'K': reconstruction['cameras'][camera]['K'], 'distortion': reconstruction['cameras'][camera]['distortion']}

    def intrinsics_from_camera(camera):
        """
        Generates intrinsics dictionary from camera dictionary.

        Args:
            camera (dict): dict({
                'width': int, (in pixels)
                'height': int, (in pixels)
                'focal': float, (normalized focal length)
                'k1': float,
                'k2': float
            })

        Returns:
            intrinsics (dict): dict({
                'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                'distortion': list([k1, k2, p1, p2, k3])
        """
        width = camera['width']
        height = camera['height']
        f = camera['focal']
        k1 = camera['k1']
        k2 = camera['k2']
        scale = max(width, height)

        K = [[f * scale, 0, width / 2], [0, f * scale, height / 2], [0, 0, 1]]
        dist = [k1, k2, 0, 0, 0]
        return {'K': K, 'dist': dist}

    for image_name in reconstruction['shots']:
        intrinsics[image_name] = intrinsics_from_camera(reconstruction['cameras'][reconstruction['shots'][image_name]['camera']])
 
    return intrinsics

import networkx as nx
    
def generate_minimal_tree(opensfm_dir):
    """
    Uses the matches to generate a minimal spanning tree of the images.
    
    """
    # Create a graph
    G = nx.Graph()

    matches_dir = os.path.join(opensfm_dir, 'matches')

    for img in  os.listdir(matches_dir):
        if not img.endswith('.pkl.gz'):
            continue
        with gzip.open(os.path.join(matches_dir, img)) as f:
            matches = pickle.load(f)

        img_name = img.split('_matches')[0]
        for target_img, features in matches.items():
            # Here, the weight is set to 1, but it could be adjusted based on feature matches
            G.add_edge(img_name, target_img, weight=1)

    # Compute the minimum spanning tree
    mst = nx.minimum_spanning_tree(G)

    # Format the output
    output = {
        "images": list(G.nodes),    
        "minimal_tree": list(mst.edges)
    }

    return output


def get_filenames(opensfm_dir):
    """
    Creates filenames.json from OpenSfM images directory.

    Args:
        opensfm_dir (str): path to OpenSfM directory

    Returns:
        filenames (dict): dict({
            'image_name': str
        })
    """
    images_dir = os.path.join(opensfm_dir, 'images')
    filenames = {}
    for filename in os.listdir(images_dir):
        if is_media_file(filename):
            filenames[filename] = filename

    return filenames