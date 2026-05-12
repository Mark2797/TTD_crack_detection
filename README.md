baseline_models
* the baseline model training and testing pipeline from the original paper with ImageNet weight

baseline_models_no_weight
* the same baseline model from the original paper but with random initialized weight

dff_HPF_multi_model
* the DFF feature extration pipeline for all the models (baseline, baseline no weight, sobelxy, sobelmaglap, and sobelmaggabor)

models_with_sobelmag_gabor
* the model training and testing pipeline with ImageNet weight, but the 3 channels are original image, sobel magnitude, gabor filter features

models_with_sobelxy
* the model training and testing pipeline with ImageNet weight, but the 3 channels are original image, sobel x direction, sobel y direction

models_with_sobelxymaglap
* the model training and testing pipeline wiht ImageNet weight, but the 3 channels are original image, sobel magnitude, Laplacian of Guassian

dataset_download.py
* code for downloading the TACK Tunnel Data (TTD) dataset from Huggingface

dff_multi_model.py
* code for multi model dff analysis

matrices.py
* performance matrices used in the original paper

prefiltering.py
* code for doing sobel filter in x, y direction, sobel magnitude with Laplacian of Guassian, and sobel magnitude with gabor filter; the 3 new channels are saved

preprocessing.py
* pipeline for the data preprocessing and mask sanitizing

preprocessing_wi_filter.py
* pipeline for the data preprocessing and mask sanitizing (the filter version)

TACK Tunnel Data (TTD).pdf
* the original TDD paper

utils.py
* some utility functions for training and testing the models