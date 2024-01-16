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
            landmark_id: match_index
        }
        img2{
            landmark_id: match_index
        }
        :
        :
        :
    }
    """
    translator = {}
    id = 0
    for matches in os.listdir(matches_dir):
        matches = load_pickle_gz(os.path.join(matches_dir, matches))

        for img in matches:
            if img not in translator:
                translator[img] = {}
            for match in matches[img]:
                if str(match[1]) not in translator[img]:
                    id += 1
                    translator[img][str(match[1])] = id

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
    main('/Users/grosche/Documents/GitHub/CVLAB/MARMOT/uploads/')
