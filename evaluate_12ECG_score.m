% This file contains functions for evaluating algorithms for the 2020 PhysioNet/
% Computing in Cardiology Challenge. You can run it as follows:
%
%   evaluate_12ECG_score(labels, outputs, scores.csv)
%
% where 'labels' is a directory containing files with the labels, 'outputs' is a
% directory containing files with the outputs from your model, and 'scores.csv'
% (optional) is a collection of scores for the algorithm outputs.
%
% Each file of labels or outputs must have the format described on the Challenge
% webpage. The scores for the algorithm outputs include the area under the
% receiver-operating characteristic curve (AUROC), the area under the recall-
% precision curve (AUPRC), accuracy (fraction of correct recordings), macro F-
% measure, and the Challenge metric, which assigns different weights to
% different misclassification errors.

function evaluate_12ECG_score(labels, outputs, output_file)
    switch nargin
        case 2
            command = ['python evaluate_12ECG_score.py' ' ' labels ' ' outputs];
        case 3
            command = ['python evaluate_12ECG_score.py' ' ' labels ' ' outputs ' ' output_file];
    end
    [status, output] = system(command);
    if status==0
        fprintf(output)
    end
end
