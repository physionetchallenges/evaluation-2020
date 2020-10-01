# PhysioNet/CinC Challenge 2020 Results

This folder contains several files with the results of the 2020 Challenge.

We introduced [new scoring metric](https://physionetchallenges.github.io/2020/#scoring) for this Challenge. We used this scoring metric to evaluate and rank the Challenge entries. We included several other metrics for reference. The area under the receiver operating characteristic (AUROC), area under the precision recall curve (AUPRC), and _F_-measure scores are the macro-average of the scores across all classes. The accuracy metric is the fraction of correctly diagnosed recordings, i.e., all classes for the recording are correct. These metrics were computed by the [evaluate_12ECG_score.py](https://github.com/physionetchallenges/evaluation-2020/blob/master/evaluate_12ECG_score.py) script in this repository. Please see this script for more details of these scores.

We included the scores on the following datasets: 
1.- Validation Set: includes recordings from CPSC and G12EC Hidden set.
2.- Hidden CPSC Set: split between the validation and test set.
3.- Hidden G12EC Set: Split between the validation and test set.
4.- Hidden Undisclosed Set: All recording were part of the test set.
5.- Test Set: Includes recording from hidden CPSC, G12EC, and undisclosed.

To refer to these tables in a publication, please cite [Perez Alday EA, Gu A, Shah A, Robichaux C, Wong A-KI, Liu C, Liu F, Bahrami Rad A, Elola A, Seyedi S, Li Q, Sharma A, Clifford GD, Reyna AR. Classification of 12-lead ECGs: the  PhysioNet/Computing in Cardiology Challenge 2020. Physiol. Meas. (In Press)](https://www.medrxiv.org/content/10.1101/2020.08.11.20172601v1).

1. Official entries that were scored on the validation and test data and ranked in the Challenge:
[physionet_2020_official_scores.csv](https://github.com/physionetchallenges/evaluation-2020/blob/master/Results/physionet_2020_official_scores.csv)
2. Unofficial entries that were scored on the validation and test data but unranked because they did not satisfy all of the [rules](https://physionetchallenges.github.io/2020/#rules-and-deadlines) or were unsuccessful on one or more of the test sets:
[physionet_2020_unofficial_scores.csv](https://github.com/physionetchallenges/evaluation-2020/blob/master/Results/physionet_2020_unofficial_scores.csv)
3. Challenge and other scoring metrics on all official entries broken with scores for each database in the validation and test data: 
[physionet_2020_full_metrics_official_entries.csv](https://github.com/physionetchallenges/evaluation-2020/blob/master/Results/physionet_2020_full_metrics_official_entries.csv )
4. Per-class scoring metrics on the validation data:
[physionet_2020_validation_metrics_by_class_official_entries.csv](https://github.com/physionetchallenges/evaluation-2020/blob/master/Results/physionet_2020_validation_metrics_by_class_official_entries.csv)
