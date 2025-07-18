# Cune-IIIF-form
Code for the OCR part of the Cune-IIIF-form project. Very much a work in progress, so nothing can be expected to work reliably all the time. Still need to figure out the ideal structure for this repo.

There are three main functionalities in this code base:
1. Data preparation
2. Training models
3. Analysis and visualization

Most code exists both for a classification and a detection task. There is no need to unify them, but it does mean that for many things there is a *classification_NAME.py* and a *detection_NAME.py* file. 

Very brief overview of the current files

* `analysis.py`: most of the classification analysis lives here
* `coco_eval.py`: evaluation code for detection, should be hidden in the future
* `condor_launcher.py`: script to launch jobs on the cluster, should not really be part of this git 
* `create_sign_map.py`: create the sign map that is necessary to convert transliterations to sign values (unicodes)
* `dataset.py`: classification dataset code, should be hidden in the future
* `detection_ds.py`: detection dataset code, should be hidden in the future
* `feature_analysis.py`: analysis and calculation of features of a model
* `multi_analysis.py`: analysis of multiple models, should check in the future of this is really necessary
* `prepare_class_data.py`: this creates the entire sign_crop datasets for classification. This is one of the few scripts that is properly cleaned and that should work well in most circumstances :).
* `test_class.py`: code for testing of classification models
* `test_detection.py`: code for testing of detection models
* `train_class.py`: train classification models
* `train_detection.py`: train detection models
* `utils.py`: a lot of utilities, should be hidden in the future
* `visualize.py`: utility to quickly visualize tablets and annotations
* `visualize_detection.py`: visualize detection of a detection model
