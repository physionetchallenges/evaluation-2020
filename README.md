# evaluation-2020
Evaluation code for the PhysioNet/Computing in Cardiology Challenge 2020

## Python

### Contents

The Python script `evaluate_12ECG_score.py` evaluate the output from your algorithm using multiple evaluation metrics that are described on the webpage for the PhysioNet/CinC Challenge 2020.

### Running

You can run the Python evaluation code by installing the NumPy Python package and running

    python evaluate_12ECG_score.py labels output scores.csv

where `labels` is a directory containing files with groundtruth labels, such as the training database on the PhysioNet webpage; `output` is a directory containing files with output produced by your algorithm; and `scores.csv` (optional) is a collection of scores for your algorithm.

## MATLAB

### Contents

The MATLAB script `evaluate_12ECG_score.m` evaluate the output from your algorithm using multiple evaluation metrics that we designed for the PhysioNet/CinC Challenge 2020.  These scripts produce the same results.  

### Running


You can run the MATLAB evaluation code by running

    evaluate_12ECG_score(labels, output, 'scores.csv')

in MATLAB, where `labels` is a directory containing files with labels, such as the training database on the PhysioNet webpage; `output` is a directory containing files with outputs produced by your algorithm; and `scores.csv` (optional) is a collection of scores for the predictions (described on the PhysioNet website).
