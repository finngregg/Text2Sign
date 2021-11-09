"""
data_loader.py:
Based on https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

Adapated from Muschick (2020) "Learn2Sign: Sign Language Recognition and Translation
using Human Keypoint Estimation and Transformer Model"
"""

import torch
from torch.utils import data
import pandas as pd
import numpy as np
import random

try:
    from run.data_utils import DataUtils
except ImportError:  
    from data_utils import DataUtils


class TextKeypointsDataset(data.Dataset):

    def __init__(self, path_to_numpy_file, path_to_csv, path_to_vocab_file, input_length, transform=None, kp_max_len=0,
                 text_max_len=0):
        self.path_to_numpy_file = path_to_numpy_file
        self.path_to_csv = path_to_csv
        self.path_to_vocab_file = path_to_vocab_file
        self.input_length = input_length
        self.transform = transform
        self.kp_max_len = kp_max_len 
        self.text_max_len = text_max_len

        # init variables
        self.int2word = {}
        self.df_kp_text_train = pd.DataFrame()

        # failes to interpret file as pickle without this
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

        # load .txt file containing keypoints and text
        self.df_kp_text_train = pd.read_csv(self.path_to_csv)

        # load keypoints
        self.saved_column_kp = self.df_kp_text_train['keypoints']
        self.all_files = np.load(self.path_to_numpy_file).item()

        # load text
        self.saved_column_text = self.df_kp_text_train['text']

        # load vocab 
        self.word2int = DataUtils().vocab_word2int(self.path_to_vocab_file) 

        # get amount of data
        self.amount_of_files = DataUtils().get_file_length(self.path_to_csv)

        # set np.load back 
        np.load = old  

    # determines the total number of data samples
    def __len__(self):
        with open(self.path_to_csv) as f:
            return sum(1 for line in f) - 1  # subtract first line (header: "keypoints,text")

    def __getitem__(self, index):
        while 1:
            # obtain subdirectory of keypoints pertaining to the index
            subdirectory = self.saved_column_kp[index]
            keys = ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']
            keys_per_folder = []

            # loop through each set of keypoints in the subdirectory
            for file in self.all_files[subdirectory]:
                temp_df = self.all_files[subdirectory][file]
                # init dictionaries & write x, y values into dictionary
                keys_x_256 = []
                keys_y_256 = []
                for k in keys:
                    if k == "pose_keypoints_2d":
                        keys_x_256 = temp_df['people'][0][k][0:25:3]
                        keys_x_256.extend(temp_df['people'][0][k][45:55:3])
                        keys_y_256 = temp_df['people'][0][k][1:26:3]
                        keys_y_256.extend(temp_df['people'][0][k][46:56:3])
                    else:
                        keys_x_256.extend(temp_df['people'][0][k][0::3])
                        keys_y_256.extend(temp_df['people'][0][k][1::3])

                keys_per_folder.append(keys_x_256 + keys_y_256 + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            # transform to tensor 
            if self.transform:
                keys_per_folder = self.transform(keys_per_folder)

            # check if padding is activated 
            # if not, break loop
            if self.kp_max_len == 0:
                break

            # break loop if the data size is smaller than the padding size
            if keys_per_folder.size(0) <= self.kp_max_len:
                break
            else:
                index = random.randint(0, self.amount_of_files - 2)

        # check if padding is activated 
        if self.kp_max_len > 0:
            if self.kp_max_len < keys_per_folder.size(0):
                temp_max_len = keys_per_folder.size(0)
            else:
                temp_max_len = self.kp_max_len
            length = keys_per_folder.size(0)
            keys = torch.zeros(temp_max_len, self.input_length)
            source = keys_per_folder
            keys[:length, :] = source

        else:
            keys = keys_per_folder

        # loads sentences from .txt file 
        sentence = []
        no_sos = []
        no_eos = []

        full_text = [int(i) for i in DataUtils().text2index([self.saved_column_text[index]], self.word2int)[0]]
        sentence.append(self.word2int["<sos>"])  # append <sos>
        sentence.extend(full_text)
        sentence.append(self.word2int["<eos>"])  # append <eos> 

        no_sos.extend(full_text)
        no_sos.append(self.word2int["<eos>"])  # append <eos> 

        no_eos.append(self.word2int["<sos>"])  # append <sos> 
        no_eos.extend(full_text)

        # Set padding length 
        padding_length = self.text_max_len
        sentence += [0] * (padding_length - len(sentence))
        no_sos += [0] * (padding_length - len(no_sos))
        no_eos += [0] * (padding_length - len(no_eos))
        if self.transform:
            sentence = self.transform(sentence)
            no_sos = self.transform(no_sos)
            no_eos = self.transform(no_eos)

        return keys, sentence, no_sos, no_eos

# convert numpy arrays to tensors
class ToTensor(object):

    def __call__(self, sample):
        return torch.from_numpy(np.array(sample))
