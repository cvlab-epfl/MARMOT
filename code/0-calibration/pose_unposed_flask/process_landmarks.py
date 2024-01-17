import json
import cv2
import pandas as pd
import numpy as np

def main():
    # create landmarks.json file
    # the output is a dictionary with keys as image names and values as a dictionary with keys 'landmarks' and 'ids'. The ids are the track ids of the landmarks in the image. The landmarks are the pixel coordinates of the landmarks in the image.
    tracks = pd.read_csv('../../data/0-calibration/opensfm/tracks.csv', delimiter='\t', skiprows=1, names=['image', 'track_id', 'feature_id', 'x', 'y', 'scale', 'r', 'g', 'b', 'segmentation', 'instance'])

    def process_group(group):
        img = cv2.imread(f'../../data/0-calibration/opensfm/images/{group.name}')
        scale_factor = max(img.shape)
        group['x'] = ((group['x'] + 0.5) * img.shape[1]).astype(int)
        group['y'] = ((group['y'] + 0.5) * img.shape[0]).astype(int)
        return {'landmarks': group[['x', 'y']].values.tolist(), 'ids': group['track_id'].values.tolist()}

    landmarks = tracks.groupby('image').apply(process_group).to_dict()

    # save landmarks to json
    with open('../../data/0-calibration/opensfm/landmarks.json', 'w') as f:
        json.dump(landmarks, f, indent=4)


    # create landmarks_global.json file
    # the output is a dictionary with keys as track ids and values x,y,z in world coordinates.
        

    # load reconstruction
    with open('../../data/0-calibration/opensfm/reconstruction.json') as f:
        reconstruction = json.load(f)[0]

    landmarks_global = {}
    landmarks_global['ids'] = []
    landmarks_global['points'] = []
    for key, value in reconstruction['points'].items():
        landmarks_global['ids'].append(key)
        landmarks_global['points'].append(value['coordinates'])

    # save landmarks_global to json
    with open('../../data/0-calibration/opensfm/landmarks_global.json', 'w') as f:
        json.dump(landmarks_global, f, indent=4)

    




if __name__ == '__main__':
    print('Generating landmarks.json and landmarks_global.json')
    main()
