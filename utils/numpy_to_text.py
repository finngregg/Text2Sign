"""
numpy_to_text.py: link the numpy files to the sentences

Adapated from Muschick (2020) "Learn2Sign: Sign Language Recognition and Translation
using Human Keypoint Estimation and Transformer Model"

"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


class CategoriesToNpy:

    def __init__(self, path_to_numpy_file, path_to_csv, path_to_target):
        self.path_to_numpy_file = Path(path_to_numpy_file)
        self.path_to_csv = Path(path_to_csv)
        self.path_to_target = Path(path_to_target)
        old = np.load
        np.load = lambda *a, **k: old(*a, **k, allow_pickle=True)

    def main(self):
        self.categories2sentence()

    def categories2sentence(self):
        # load keypoints from numoy file
        kp_files = np.load(self.path_to_numpy_file).item()
        df_kp = pd.DataFrame(kp_files.keys(), columns=["keypoints"])
        kp2sentence = []

        d = {'keypoints': [], 'text': []}
        # load sentences from .txt file
        with open(self.path_to_csv) as f:
            for line in f:
                d['keypoints'].append(line.split(" ")[0])
                d['text'].append(" ".join(line.split()[1:]))
        df_text = pd.DataFrame(d)

        speaker = []
        counter = 0
        # loop through frames of keypoints in the numpy file
        for kp in df_kp["keypoints"]:
            vid_speaker = kp[:11]
            speaker.append(vid_speaker)
            # loops through corresponding setences IDs of keypoints
            for idx in range(len(df_text['keypoints'])):
                # checks sentence ID is in the .txt file
                if vid_speaker in df_text['keypoints'][idx]:
                    kp2sentence.append([kp, df_text['text'][idx]])
                    break

            if counter % 250 == 0:
                print("Folder %d of %d" % (counter, len(df_kp["keypoints"])))
            counter += 1
        df_kp_text_train = pd.DataFrame(kp2sentence, columns=["keypoints", "text"])
        # store updated data to new .txt file
        df_kp_text_train.to_csv(self.path_to_target / str(str(self.path_to_csv.name) + "_2npy.txt"), index=False)


if __name__ == '__main__':
    # numpy file containing normalized keypoints
    if len(sys.argv) > 1:
        path_to_numpy_file = sys.argv[1]
    else:
        print("Set path to npy file")
        sys.exit()

    # .txt file containing transformed sentences 
    if len(sys.argv) > 2:
        path_to_csv = sys.argv[2]
    else:
        print("Set path to transformed file containing categories")
        sys.exit()

    # target directory
    if len(sys.argv) > 3:
        path_to_target = sys.argv[3]
    else:
        print("Set path to target folder")
        sys.exit()

    npy = CategoriesToNpy(path_to_numpy_file, path_to_csv, path_to_target)
    npy.main()