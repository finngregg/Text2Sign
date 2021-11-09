"""keypoint_normalization.py: normalize over all files from each folder of an directory

run using the numpy files for training, validation and testing 

Adapated from Muschick (2020) "Learn2Sign: Sign Language Recognition and Translation
using Human Keypoint Estimation and Transformer Model"

"""

import json
import random
import numpy as np
import os
import statistics
from pathlib import Path
import sys
import time
import warnings
import os
import psutil
import copy


class Normalize:

    def __init__(self, path_to_numpy_file, path_to_target_dir="", path_to_json_dir=""):
        self.path_to_numpy_file = Path(path_to_numpy_file)
        self.path_to_target_dir = path_to_target_dir
        self.path_to_json = Path(path_to_json_dir)
        self.keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']

        # failes to interpret file as pickle without this
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def main(self):

        # create target directory
        self.create_folders()

        # do not centralize values 
        all_files_dictionary_centralized = None

        # normalize values
        all_mean_stdev = self.compute_mean_stdev_transposed(all_files_dictionary_centralized)
        self.normalize(all_mean_stdev, all_files_dictionary_centralized)

    def create_folders(self):
        # name target directory
        data_dir_target = self.path_to_numpy_file.parent

        # set target directory
        self.path_to_target_dir = Path(data_dir_target)

    # read data from numpy file and compute mean and stdev for each frame of keypoints
    def compute_mean_stdev_transposed(self, all_files_dictionary_centralized=None):
        
        all_files = self.dictionary_check(all_files_dictionary_centralized)
        # empty dictionary to store means and stdev
        all_mean_stdev = {} 
        once = 1
        all_files_xy = {'all': {}}

        for subdir in all_files.keys():
            # load files from one folder into dictionary
            for file in all_files[subdir]:
                temp_df = all_files[subdir][file]
                if once == 1:
                    for k in self.keys:
                        all_files_xy['all'][k] = {'x': [], 'y': []}
                    once = 0
                for k in self.keys:
                    all_files_xy['all'][k]['x'].append(temp_df['people'][0][k][0::3])
                    all_files_xy['all'][k]['y'].append(temp_df['people'][0][k][1::3])

        print("Files read, computing mean and stdev")
        for k in self.keys:
            mean_stdev_x = []
            mean_stdev_y = []

            for list in np.array(all_files_xy['all'][k]['x']).T.tolist():
                
                if "Null" in list:
                    list = [i for i in list if i != "Null"]
                    if list == []:
                        mean_stdev_x.append(["Null", "Null"])
                    else:
                        list = [float(item) for item in list]
                        mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_x.append([np.mean(list), statistics.pstdev(list)])

            for list in np.array(all_files_xy['all'][k]['y']).T.tolist():
                if "Null" in list:
                    list = [i for i in list if i != "Null"]
                    if list == []:
                        mean_stdev_y.append(["Null", "Null"])
                    else:
                        list = [float(item) for item in list]
                        mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])
                else:
                    list = [float(item) for item in list]
                    mean_stdev_y.append([np.mean(list), statistics.pstdev(list)])

            all_mean_stdev[k] = [np.array(mean_stdev_x).T.tolist(), np.array(mean_stdev_y).T.tolist()]

        # write the means and std_dev into json file
        f = open(self.path_to_target_dir / "all_mean_stdev.json", "w")
        f.write(json.dumps(all_mean_stdev))
        f.close()

        return all_mean_stdev

    def normalize(self, all_mean_stdev, all_files_dictionary_centralized=None):

        all_files = self.dictionary_check(all_files_dictionary_centralized)

        all_files_save = {}
        # use mean and stdev to compute values for the json files
        for subdir in all_files.keys():
            all_files_save[subdir] = {}
            for file in all_files[subdir]:
                data = all_files[subdir][file]

                # x component -> [0::3]
                # y component -> [1:.3]
                # c confidence score -> [2::3]
                for k in self.keys:
                    
                    temp_x = data['people'][0][k][0::3]
                    temp_y = data['people'][0][k][1::3]
                    temp_c = data['people'][0][k][2::3]

                    # get x values and normalize it
                    for index in range(len(temp_x)):
                        mean_x = all_mean_stdev[k][0][0][index]
                        stdev_x = all_mean_stdev[k][0][1][index]

                        mean_y = all_mean_stdev[k][1][0][index]
                        stdev_y = all_mean_stdev[k][1][1][index]

                        if temp_x[index] == "Null":
                            temp_x[index] = temp_x[index]
                        elif str(stdev_x) == "Null":
                            temp_x[index] = temp_x[index]
                        elif float(stdev_x) == 0:
                            temp_x[index] = temp_x[index]
                        else:
                            temp_x[index] = (temp_x[index] - float(mean_x)) / float(stdev_x)

                        if temp_y[index] == "Null":
                            temp_y[index] = temp_y[index]
                        elif str(stdev_y) == "Null":
                            temp_y[index] = temp_y[index]
                        elif float(stdev_y) == 0:
                            temp_y[index] = temp_y[index]
                        else:
                            temp_y[index] = (temp_y[index] - float(mean_y)) / float(stdev_y)

                    # new array to store normalized values
                    values = []
                    for index in range(len(temp_x)):
                        values.append(temp_x[index])
                        values.append(temp_y[index])
                        values.append(temp_c[index])

                    # copy the new array into its original storage
                    data['people'][0][k] = values

                all_files_save[subdir][file] = data
       
        dictionary_file_path = self.path_to_target_dir / 'all_files_normalized.npy'
        last_folder = os.path.basename(os.path.normpath(dictionary_file_path.parent)) + "/" + str(
            dictionary_file_path.name)
        print("Saving normalized results to %s " % last_folder)
        np.save(dictionary_file_path, all_files_save)

    def dictionary_check(self, all_files_dictionary_centralized):
        # load from numpy file
        if all_files_dictionary_centralized is None:
            print("Loading from %s file" % self.path_to_numpy_file)
            all_files = np.load(self.path_to_numpy_file).item()
        else:
            print("Using internal centralized dictionary")
            all_files = all_files_dictionary_centralized
        return all_files

    def save_dictionary_to_file(self, subdirectories):
        dictionary_file_path = self.path_to_target_dir / 'all_files.npy'
        last_folder = os.path.basename(os.path.normpath(dictionary_file_path.parent)) + "/" + str(
            dictionary_file_path.name)

        if dictionary_file_path.is_file():
            print(".../%s file already exists. Not copying files " % last_folder)
            return dictionary_file_path
        else:
            print("Saving files to %s " % dictionary_file_path)

        # use keys of openpose here
        all_files = {}

        for subdir in subdirectories:
            print("Reading files from %s" % subdir)
            json_files = [pos_json for pos_json in os.listdir(Path(self.path_to_json) / subdir)
                          if pos_json.endswith('.json')]
            all_files[subdir] = {}
            # load files from one folder into dictionary
            for file in json_files:
                temp_df = json.load(open(Path(self.path_to_json) / subdir / file))
                all_files[subdir][file] = temp_df

        np.save(dictionary_file_path, all_files)
        return Path(dictionary_file_path)

if __name__ == '__main__':
    # path to numpy file
    if len(sys.argv) > 1:
        path_to_numpy_file = sys.argv[1]
    else:
        print("Set numpy file")
        sys.exit()

    # directories
    path_to_target_dir = ""
    path_to_json_dir = ""

    norm = Normalize(path_to_numpy_file, path_to_target_dir, path_to_json_dir)
    norm.main()