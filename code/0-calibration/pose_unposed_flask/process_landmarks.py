import pickle
import gzip
import numpy as np
import os
import json

def load_pickle_gz(filename):
    print(filename)
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object
    

def matches_translator(matches_dir):
    """
    Creates a unique id for each landmark , returns dictionary:
    dict{
        img1{
            match_index: landmark_id
        }
        img2{
            match_index: landmark_id
        }
        :
        :
        :
    }
    """
    translator = {}
    id = 0
    for matches_file in os.listdir(matches_dir):
        matches = load_pickle_gz(os.path.join(matches_dir, matches_file))
        img1 = matches_file.split('_')[0]
        if img1 not in translator:
            translator[img1] = {}
        for img in matches:
            if img not in translator:
                translator[img] = {}
            for match in matches[img]:
                added = added_1 = False
                if str(match[1]) not in translator[img]:
                    added = True
                    translator[img][str(match[1])] = id
                if str(match[0]) not in translator[img1]:
                    added_1 = True
                    translator[img][str(match[0])] = id
                if added and added_1:
                    id += 2
                
            # sort matches by landmark id
            translator[img] = dict(sorted(translator[img].items(), key=lambda item: item[1]))

    return translator
    

def main(upload_folder):
    match_path = os.path.join(upload_folder, 'matches')
    # print(match_path)
    translation = matches_translator(match_path)
    # save translation
    with open(os.path.join(upload_folder, 'translation.json'), 'w') as f:
        json.dump(translation, f)

    # landmarks = {}
    # landmarks_global = {}
    # img_path = os.path.join()

    # for img1 in os.iterdir(img_path):
    #     match_path = os.path.join('matches', img1, '_matches.pkl.gz')
    #     features_path = os.path.join('features', img1, '.features.npz')
    #     matches = load_pickle_gz(match_path)
    #     features = np.load(features_path)
    #     translator = {}
    #     id = 0
    #     for img2 in matches:
    #         if img2 not in landmarks:
    #             landmarks[img2] = {}
    #         for match in matches[img2]:
    #             translator[id] = (match[0], match[1])
    #             landmarks[img2]["ids"]
    #             .append(features[match[0]])
    #             id += 1

        




if __name__ == '__main__':
    print('Running process_landmarks.py')
    main('F:/Github/MARMOT/uploads')
