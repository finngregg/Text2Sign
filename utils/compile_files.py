"""compile_files.py: compile json files in a single numpy file

run using the json file directories for training, validation and testing 

Adapated from Muschick (2020) "Learn2Sign: Sign Language Recognition and Translation
using Human Keypoint Estimation and Transformer Model"

"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import psutil


class SaveFiles:

    def __init__(self, path_to_json_dir):
        self.path_to_json = Path(path_to_json_dir)

        # keys for json file 
        self.keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
    
        # failes to interpret file as pickle without this
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def create_folders(self):
        # name new target directory
        data_dir_target = self.path_to_json.parent / str(self.path_to_json.name + "_saved_numpy")

        # create new target directory
        if not os.path.exists(data_dir_target):
            os.makedirs(data_dir_target)

        subdirectories = [x[1] for x in os.walk(self.path_to_json)][0]

        print("%d folders left." % len(subdirectories))
        if len(subdirectories) == 0:
            print("No subdirectories left. Exit.")
            sys.exit()

        return data_dir_target, subdirectories

    def copy_dictionary_to_file(self, data_dir_target, subdirectories):
        dictionary_file_path = data_dir_target / 'raw_data.npy'
        subdirectories_file = subdirectories.copy()

        print("Saving files to %s " % dictionary_file_path)

        all_files = {}
        index = 0

        for subdir in subdirectories:
            index += 1
            print("%d of %d" % (index, len(subdirectories)))
            print("Reading files from %s" % subdir)

            json_files = [pos_json for pos_json in os.listdir(self.path_to_json / subdir)
                          if pos_json.endswith('.json')]
            all_files[subdir] = {}
            # loads files from subdirectory e.g. -fZc293MpJk_0-1-rgb_front
            for file in json_files:
                temp_df = json.load(open(self.path_to_json / subdir / file))
                all_files[subdir][file] = temp_df

            subdirectories_file.remove(subdir)

        # update numpy file
        if os.path.isfile(dictionary_file_path):
            dictionary_from_file = np.load(dictionary_file_path).item()
            dictionary_from_file.update(all_files)
            np.save(dictionary_file_path, dictionary_from_file)
        else:
            np.save(dictionary_file_path, all_files)

        return Path(dictionary_file_path)

    def main(self):
        # create folders and paths
        data_dir_target, subdirectories = self.create_folders()

        # compile files into numpy and store in target directory
        self.copy_dictionary_to_file(data_dir_target, subdirectories)


if __name__ == '__main__':
    # json files directory
    if len(sys.argv) > 1:
        path_to_json_dir = sys.argv[1]
    else:
        print("Set json file directory")
        sys.exit()

    try:
        norm = SaveFiles(path_to_json_dir)
        norm.main()
    except NameError:
        print("Set paths")