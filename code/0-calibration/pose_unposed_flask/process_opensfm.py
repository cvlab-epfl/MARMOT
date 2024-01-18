import json
import cv2
import pandas as pd
import os

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
        landmarks_global['ids'].append(key)
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


        for camera in reconstruction['cameras']:
            intrinsics[camera['image']] = {'K': camera['K'], 'distortion': camera['distortion']}

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
    



def main():
    # set path to opensfm directory containing reconstruction.json and tracks.csv
    opensfm_dir = os.path.join('..', '..', 'data', '0-calibration', 'opensfm', 'undistorted')

    # create landmarks.json file at landmarks_path
    landmarks = get_landmarks(opensfm_dir)

    landmarks_path = os.path.join(opensfm_dir, 'landmarks.json')
    with open(landmarks_path, 'w') as f:
        json.dump(landmarks, f, indent=4)


    # create landmarks_global.json file at landmarks_global_path
    landmarks_global = get_landmarks_global(opensfm_dir)
    
    landmarks_global_path = os.path.join(opensfm_dir, 'landmarks_global.json')
    with open(landmarks_global_path, 'w') as f:
        json.dump(landmarks_global, f, indent=4)

    # create intrinsics.json file at intrinsics_path
    calibs_dir = os.path.join('..', '..', 'data', '0-calibration', 'calibs')
    intrinsics = get_intrinsics(calibs_dir, opensfm_dir)

    intrinsics_path = os.path.join(opensfm_dir, 'intrinsics.json')
    with open(intrinsics_path, 'w') as f:
        json.dump(intrinsics, f, indent=4)

if __name__ == '__main__':
    print('Generating landmarks.json and landmarks_global.json')
    main()
