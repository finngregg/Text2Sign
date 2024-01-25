# README

This repository contains the code required to translate English text to for sign language using 2D keypoint estimation and the How2Sign ASL dataset.

## Requirements

### How2Sign
- How2Sign: B-F-H 2D Keypoints clips for training, validation and testing
- How2Sign: English translations for training, validation and testing
- Python (3.6.9), Pytorch (1.5.0), pytorch-lightning (0.7.3)

## Usage
- Create virtual environment using the command line: 
    - cd .virtualenvs
    - virtualenv env
    - source venv/bin/activate
- To deactivate
    - deactivate 

- Install requirements.txt
    - pip install -r ../requirements.txt

## Utils
- compile_files.py
    - compile multiple JSON files into one numpy file
    - python3 ../utils/compile_files.py "path to directory containing folders of JSON files" 
    - example directory: json_train -> sentence ID e.g. -fZc293MpJk_0-1-rgb_front ->  -fZc293MpJk_0-1-rgb_front_keypoints_0000

- data_normalization.py 
    - applies 2D object normalization to keypoint data
    - python3 ../utils/data_normalization.py "path to numpy file" 

- print_numpy.py
    - ouput the contents of a numpy file 

- text_utils.py
    - tokenize, normalize and clean text 
    - create vocabulary
    - python3 ../utils/text_utils.py "path to .csv file containing English sentences"

- numpy_to_text.py
    - link numpy files to sentences
    - python3 ../utils/numpy_to_text.py "path to normalized numpy file" "path to .txt file containing transformed sentences" "target directory"

## Run 

- main.py
    - run main application
    - python3 ../run/main.py ../run/hparams.json

- hparam.json
    - input_size: 256 for How2Sign dataset
    - output_size: length of vocab.txt file
