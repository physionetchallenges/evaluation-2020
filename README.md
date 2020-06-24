# PhysioNet/CinC Challenge 2020 Evaluation Metrics

This repository contains the Python and MATLAB evaluation code for the PhysioNet/Computing in Cardiology Challenge 2020. The `evaluate_12ECG_score` script evaluates the output of your algorithm using the evaluation metrics that are described on the [webpage](https://physionetchallenges.github.io/2020/) for the PhysioNet/CinC Challenge 2020.

You can run the Python evaluation code by installing the NumPy Python package and running

    python evaluate_12ECG_score.py labels outputs scores.csv

where `labels` is a directory containing files with one or more labels for each 12-lead ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; and `scores.csv` (optional) is a collection of scores for your algorithm.
