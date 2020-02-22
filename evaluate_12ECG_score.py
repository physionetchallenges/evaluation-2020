#!/usr/bin/env python

# This file contains functions for evaluating algorithms for the 2020 PhysioNet/
# CinC Challenge. You can run it as follows:
#
#   python evaluate_12ECG_score.py labels output scores.csv
#
# where 'labels' is a directory containing files with labels, 'output' is a
# directory containing files with output labels from your model, a
# and 'scores.csv' (optional) is a collection of scores for the output.

################################################################################

# The evaluate_scores function computes a Fbeta measure and a generalizatoin of
# the Jaccard measure but giving missed diagnosis twice as much weight as
# correct diagnoses and false alarms
#
# Inputs:
#   'label_directory' is a input directory with the header files and #Dx true labels
#
#   'output_directory' is a directory of comma-delimited text files, where
#   the first row of the file is the output label for each class and
#   the second row of the file is the probability of the class label. 
#   Note that there must be an output/value for every label.
#
# Outputs:
#
#   'fbeta_measure' is Fbeta-measure, with beta = 2
#
#   'Gbeta_score' is a generalization of the Jaccard measures but giving missed 
#   diagnoses twice as much weight as correct diagnoses and false alarms, beta = 2
#
#   'accuracy' is accuracy.
#
#   'f_measure' is F-measure.
#
#
# Example:
#   Omitted due to length. See the below examples.

import numpy as np, os, os.path, sys


def evaluate_12ECG_score(label_directory, output_directory):

    # Set parameters.
    label_header       = '12ECGLabel'
    output_label_header  = 'OutputLabel'
    output_probability_header = 'OutputProbability'

    beta = 2
    labels=[]
    output=[]
    output_probabilities=[]

    # Find label and output files.
    label_files = []
    for f in os.listdir(label_directory):
        g = os.path.join(label_directory, f)
        if os.path.isfile(g) and not f.lower().startswith('.') and f.lower().endswith('hea'):
            label_files.append(g)
    label_files = sorted(label_files)

    output_files = []
    for f in os.listdir(output_directory):
        g = os.path.join(output_directory, f)
        if os.path.isfile(g) and not f.lower().startswith('.') and f.lower().endswith('csv'):
            output_files.append(g)
    output_files = sorted(output_files)

    if len(label_files) != len(output_files):
        raise Exception('Numbers of label and output files must be the same.')

    classes = get_classes(label_files)


    # Load labels and outputs.
    num_files = len(label_files)

    for k in range(num_files):

        recording_label,classes_label,single_recording_labels=get_true_labels(label_files[k],classes)
        
        with open(output_files[k],'r') as f:
            tmp_data = f.readlines()
        recording_output = tmp_data[0]
        classes_output = tmp_data[1].split(',')
        single_recording_output = np.array(tmp_data[2].split(','),np.int)
        single_probabilities_output = np.array(tmp_data[3].split(','),np.float64)

       # Check labels and output for errors.

        if not (len(classes_label) == len(classes_output)):
            raise Exception('Numbers of classes for a file must be the same.')
        
        if not (len(single_recording_labels) == len(single_recording_output) == len(single_probabilities_output)):
            raise Exception('Numbers of labels and output for a file must be the same.')

        labels.append(single_recording_labels)
        output.append(single_recording_output)
        output_probabilities.append(single_probabilities_output)

    labels=np.array(labels)
    output=np.array(output)
    output_probabilities=np.array(output_probabilities)

    num_classes = len(classes_label)

    # Compute F_beta measure and the generalization of the Jaccard index
    accuracy,f_measure,Fbeta_measure,Gbeta_measure = compute_beta_score(labels, output, beta, num_classes)

    # compute AUROC and AUPRC
    auroc,auprc = compute_auc(labels, output_probabilities,num_classes)

    return auroc,auprc,accuracy,f_measure,Fbeta_measure,Gbeta_measure


# Find unique number of classes
def get_classes(files):

    classes=set()
    for input_file in files:
        with open(input_file,'r') as f:
            for lines in f:
                if lines.startswith('#Dx'):
                    tmp = lines.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())

    return sorted(classes)



# Find unique true labels
def get_true_labels(input_file,classes):

    classes_label = classes
    single_recording_labels=np.zeros(len(classes),dtype=int)


    with open(input_file,'r') as f:
        first_line = f.readline()
        recording_label=first_line.split(' ')[0]
        print(recording_label)
        for lines in f:
            if lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    idx = classes.index(c.strip())
                    single_recording_labels[idx]=1

    return recording_label,classes_label,single_recording_labels





# The compute_beta_score function computes the Fbeta-measure given an specific beta value
# and the G value define at the begining of the file.
#
# Inputs:
#   'labels' are the true classes of the recording
#
#   'output' are the output classes of your model
#
#   'beta' is the weight
#
# Outputs:
#
# fbeta_measure, Fbeta measure given an specific beta
# Gbeta_measure, Generalization of the Jaccard measure with a beta weigth
#

def compute_beta_score(labels, output, beta, num_classes, check_errors=True):

    # Check inputs for errors.
    if check_errors:
        if len(output) != len(labels):
            raise Exception('Numbers of outputs and labels must be the same.')

    # Populate contingency table.
    num_recordings = len(labels)

    fbeta_l = np.zeros(num_classes)
    gbeta_l = np.zeros(num_classes)
    fmeasure_l = np.zeros(num_classes)
    accuracy_l = np.zeros(num_classes)

    f_beta = 0
    g_beta = 0
    f_measure = 0
    accuracy = 0

    # Weight function
    C_l=np.ones(num_classes);

    for j in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        tn = 0

        for i in range(num_recordings):
            
            num_labels = np.sum(labels[i])
        
            if labels[i][j] and output[i][j]:
                tp += 1/num_labels
            elif not labels[i][j] and output[i][j]:
                fp += 1/num_labels
            elif labels[i][j] and not output[i][j]:
                fn += 1/num_labels
            elif not labels[i][j] and not output[i][j]:
                tn += 1/num_labels

        # Summarize contingency table.
        if ((1+beta**2)*tp + (fn*beta**2) + fp):
            fbeta_l[j] = float((1+beta**2)* tp) / float(((1+beta**2)*tp) + (fn*beta**2) + fp)
        else:
            fbeta_l[j] = 1.0

        if (tp + fp + beta * fn):
            gbeta_l[j] = float(tp) / float(tp + fp + beta*fn)
        else:
            gbeta_l[j] = 1.0

        if tp + fp + fn + tn:
            accuracy_l[j] = float(tp + tn) / float(tp + fp + fn + tn)
        else:
            accuracy_l[j] = 1.0

        if 2 * tp + fp + fn:
            fmeasure_l[j] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            fmeasure_l[j] = 1.0


    for i in range(num_classes):
        f_beta += fbeta_l[i]*C_l[i]
        g_beta += gbeta_l[i]*C_l[i]
        f_measure += fmeasure_l[i]*C_l[i]
        accuracy += accuracy_l[i]*C_l[i]


    f_beta = float(f_beta)/float(num_classes)
    g_beta = float(g_beta)/float(num_classes)
    f_measure = float(f_measure)/float(num_classes)
    accuracy = float(accuracy)/float(num_classes)


    return accuracy,f_measure,f_beta,g_beta

    
# The compute_auc function computes AUROC and AUPRC as well as other summary
# statistics (TP, FP, FN, TN, TPR, TNR, PPV, NPV, etc.) that can be exposed
# from this function.
#
# Inputs:
#   'labels' are the true classes of the recording
#
#   'output' are the output classes of your model
#
#   'beta' is the weight
#
#
# Outputs:
#   'auroc' is a scalar that gives the AUROC of the algorithm using its
#   output probabilities, where specificity is interpolated for intermediate
#   sensitivity values.
#
#   'auprc' is a scalar that gives the AUPRC of the algorithm using its
#   output probabilities, where precision is a piecewise constant function of
#   recall.
#



def compute_auc(labels, probabilities, num_classes, check_errors=True):


    # Check inputs for errors.
    if check_errors:
        if len(labels) != len(probabilities):
            raise Exception('Numbers of outputs and labels must be the same.')

    auroc_l = np.zeros(num_classes)
    auprc_l = np.zeros(num_classes)

    auroc = 0
    auprc = 0

    # Weight function - this will change
    C_l=np.ones(num_classes);

    # Populate contingency table.
    num_recordings = len(labels)

    for k in range(num_classes):
    

            # Find probabilities thresholds.
        thresholds = np.unique(probabilities[:,k])[::-1]
        if thresholds[0] != 1:
            thresholds = np.insert(thresholds, 0, 1)
        if thresholds[-1] == 0:
            thresholds = thresholds[:-1]

        m = len(thresholds)
    

        # Populate contingency table across probabilities thresholds.
        tp = np.zeros(m)
        fp = np.zeros(m)
        fn = np.zeros(m)
        tn = np.zeros(m)

        # Find indices that sort the predicted probabilities from largest to
        # smallest.
        idx = np.argsort(probabilities[:,k])[::-1]

        i = 0
        for j in range(m):
            # Initialize contingency table for j-th probabilities threshold.
            if j == 0:
                tp[j] = 0
                fp[j] = 0
                fn[j] = np.sum(labels[:,k])
                tn[j] = num_recordings - fn[j]
            else:
                tp[j] = tp[j - 1]
                fp[j] = fp[j - 1]
                fn[j] = fn[j - 1]
                tn[j] = tn[j - 1]
            # Update contingency table for i-th largest predicted probability.
            while i < num_recordings and probabilities[idx[i],k] >= thresholds[j]:
                if labels[idx[i],k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize contingency table.
        tpr = np.zeros(m)
        tnr = np.zeros(m)
        ppv = np.zeros(m)
        npv = np.zeros(m)


        for j in range(m):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = 1
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = 1
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = 1
            if fn[j] + tn[j]:
                npv[j] = float(tn[j]) / float(fn[j] + tn[j])
            else:
                npv[j] = 1

        # Compute AUROC as the area under a piecewise linear function with TPR /
        # sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR / recall (x-axis) and PPV / precision
        # (y-axis).

        for j in range(m-1):
            auroc_l[k] += 0.5 * (tpr[j + 1] - tpr[j]) * (tnr[j + 1] + tnr[j])
            auprc_l[k] += (tpr[j + 1] - tpr[j]) * ppv[j + 1]


    for i in range(num_classes):
        auroc += auroc_l[i]*C_l[i]
        auprc += auprc_l[i]*C_l[i]

    auroc = float(auroc)/float(num_classes)
    auprc = float(auprc)/float(num_classes)

    
    return auroc, auprc



if __name__ == '__main__':

    auroc,auprc,accuracy,f_measure,f_beta,g_beta = evaluate_12ECG_score(sys.argv[1], sys.argv[2])

    output_string = 'AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure\n{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}'.format(auroc,auprc,accuracy,f_measure,f_beta,g_beta)
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
    else:
        print(output_string)
