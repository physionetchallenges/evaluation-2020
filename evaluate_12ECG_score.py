#!/usr/bin/env python

# This file contains functions for evaluating algorithms for the 2020 PhysioNet/
# Computing in Cardiology Challenge. You can run it as follows:
#
#   python evaluate_12ECG_score.py labels outputs scores.csv
#
# where 'labels' is a directory containing files with the labels, 'outputs' is a
# directory containing files with the outputs from your model, and 'scores.csv'
# (optional) is a collection of scores for the algorithm outputs.
#
# Each file of labels or outputs must have the format described on the Challenge
# webpage. The scores for the algorithm outputs include the area under the
# receiver-operating characteristic curve (AUROC), the area under the recall-
# precision curve (AUPRC), accuracy (fraction of correct recordings), macro F-
# measure, and the Challenge metric, which assigns different weights to
# different misclassification errors.

import numpy as np, os, os.path, sys

def evaluate_12ECG_score(label_directory, output_directory):
    # Define the weights, the SNOMED CT code for the normal class, and equivalent SNOMED CT codes.
    weights_file = 'weights.csv'
    normal_class = '426783006'
    equivalent_classes = [['713427006', '59118001'], ['284470004', '63593006'], ['427172004', '17338001']]

    # Find the label and output files.
    print('Finding label and output files...')
    label_files, output_files = find_challenge_files(label_directory, output_directory)

    # Load the labels and outputs.
    print('Loading labels and outputs...')
    label_classes, labels = load_labels(label_files, normal_class, equivalent_classes)
    output_classes, binary_outputs, scalar_outputs = load_outputs(output_files, normal_class, equivalent_classes)

    # Organize/sort the labels and outputs.
    print('Organizing labels and outputs...')
    classes, labels, binary_outputs, scalar_outputs = organize_labels_outputs(label_classes, output_classes, labels, binary_outputs, scalar_outputs)

    # Load the weights for the Challenge metric.
    print('Loading weights...')
    weights = load_weights(weights_file, classes)

    # Evaluate the model by comparing the labels and outputs.
    print('Evaluating model...')

    print('- AUROC and AUPRC...')
    auroc, auprc = compute_auc(labels, scalar_outputs)

    print('- Accuracy...')
    accuracy = compute_accuracy(labels, binary_outputs)

    print('- F-measure...')
    f_measure = compute_f_measure(labels, binary_outputs)

    print('- F-beta and G-beta measures...')
    f_beta_measure, g_beta_measure = compute_beta_measures(labels, binary_outputs, beta=2)

    print('- Challenge metric...')
    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, normal_class)

    print('Done.')

    # Return the results.
    return auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric

# Check if the input is a number.
def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

# Find Challenge files.
def find_challenge_files(label_directory, output_directory):
    label_files = list()
    output_files = list()
    for f in sorted(os.listdir(label_directory)):
        F = os.path.join(label_directory, f) # Full path for label file
        if os.path.isfile(F) and F.lower().endswith('.hea') and not f.lower().startswith('.'):
            root, ext = os.path.splitext(f)
            g = root + '.csv'
            G = os.path.join(output_directory, g) # Full path for corresponding output file
            if os.path.isfile(G):
                label_files.append(F)
                output_files.append(G)
            else:
                raise IOError('Output file {} not found for label file {}.'.format(g, f))

    if label_files and output_files:
        return label_files, output_files
    else:
        raise IOError('No label or output files found.')

# Load labels from header/label files.
def load_labels(label_files, normal_class, equivalent_classes_collection):
    # The labels should have the following form:
    #
    # Dx: label_1, label_2, label_3
    #
    num_recordings = len(label_files)

    # Load diagnoses.
    tmp_labels = list()
    for i in range(num_recordings):
        with open(label_files[i], 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    dxs = set(arr.strip() for arr in l.split(': ')[1].split(','))
                    tmp_labels.append(dxs)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print('- The normal class {} is not one of the label classes, so it has been automatically added, but please check that you chose the correct normal class.'.format(normal_class))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for labels.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for dx in dxs:
            j = classes.index(dx)
            labels[i, j] = 1

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The label for the representative class is positive if any of the labels in the set is positive.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            labels[:, representative_index] = np.any(labels[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    labels = np.delete(labels, remove_indices, axis=1)

    # If the labels for the normal class and one or more other classes are positive, then make the label for the normal class negative.
    # If the labels for all classes are negative, then make the label for the normal class positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(labels[i, :])
        if labels[i, normal_index]==1 and num_positive_classes>1:
            labels[i, normal_index] = 0
        elif num_positive_classes==0:
            labels[i, normal_index] = 1

    return classes, labels

# Load outputs from output files.
def load_outputs(output_files, normal_class, equivalent_classes_collection):
    # The outputs should have the following form:
    #
    # diagnosis_1, diagnosis_2, diagnosis_3
    #           0,           1,           1
    #        0.12,        0.34,        0.56
    #
    num_recordings = len(output_files)

    tmp_labels = list()
    tmp_binary_outputs = list()
    tmp_scalar_outputs = list()
    for i in range(num_recordings):
        with open(output_files[i], 'r') as f:
            for j, l in enumerate(f):
                arrs = [arr.strip() for arr in l.split(',')]
                if j==1:
                    row = arrs
                    tmp_labels.append(row)
                elif j==2:
                    row = list()
                    for arr in arrs:
                        number = 1 if arr in ('1', 'True', 'true', 'T', 't') else 0
                        row.append(number)
                    tmp_binary_outputs.append(row)
                elif j==3:
                    row = list()
                    for arr in arrs:
                        number = float(arr) if is_number(arr) else 0
                        row.append(number)
                    tmp_scalar_outputs.append(row)

    # Identify classes.
    classes = set.union(*map(set, tmp_labels))
    if normal_class not in classes:
        classes.add(normal_class)
        print('- The normal class {} is not one of the output classes, so it has been automatically added, but please check that you identified the correct normal class.'.format(normal_class))
    classes = sorted(classes)
    num_classes = len(classes)

    # Use one-hot encoding for binary outputs and the same order for scalar outputs.
    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for i in range(num_recordings):
        dxs = tmp_labels[i]
        for k, dx in enumerate(dxs):
            j = classes.index(dx)
            binary_outputs[i, j] = tmp_binary_outputs[i][k]
            scalar_outputs[i, j] = tmp_scalar_outputs[i][k]

    # For each set of equivalent class, use only one class as the representative class for the set and discard the other classes in the set.
    # The binary output for the representative class is positive if any of the classes in the set is positive.
    # The scalar output is the mean of the scalar outputs for the classes in the set.
    remove_classes = list()
    remove_indices = list()
    for equivalent_classes in equivalent_classes_collection:
        equivalent_classes = [x for x in equivalent_classes if x in classes]
        if len(equivalent_classes)>1:
            representative_class = equivalent_classes[0]
            other_classes = equivalent_classes[1:]
            equivalent_indices = [classes.index(x) for x in equivalent_classes]
            representative_index = equivalent_indices[0]
            other_indices = equivalent_indices[1:]

            binary_outputs[:, representative_index] = np.any(binary_outputs[:, equivalent_indices], axis=1)
            scalar_outputs[:, representative_index] = np.nanmean(scalar_outputs[:, equivalent_indices], axis=1)
            remove_classes += other_classes
            remove_indices += other_indices

    for x in remove_classes:
        classes.remove(x)
    binary_outputs = np.delete(binary_outputs, remove_indices, axis=1)
    scalar_outputs = np.delete(scalar_outputs, remove_indices, axis=1)

    # If any of the outputs is a NaN, then replace it with a zero.
    binary_outputs[np.isnan(binary_outputs)] = 0
    scalar_outputs[np.isnan(scalar_outputs)] = 0

    # If the binary outputs for the normal class and one or more other classes are positive, then make the binary output for the normal class negative.
    # If the binary outputs for all classes are negative, then make the binary output for the normal class positive.
    normal_index = classes.index(normal_class)
    for i in range(num_recordings):
        num_positive_classes = np.sum(binary_outputs[i, :])
        if binary_outputs[i, normal_index]==1 and num_positive_classes>1:
            binary_outputs[i, normal_index] = 0
        elif num_positive_classes==0:
            binary_outputs[i, normal_index] = 1

    return classes, binary_outputs, scalar_outputs

# Organize labels and outputs.
def organize_labels_outputs(label_classes, output_classes, tmp_labels, tmp_binary_outputs, tmp_scalar_outputs):
    # Include all classes from either the labels or the outputs.
    classes = sorted(set(label_classes) | set(output_classes))
    num_classes = len(classes)

    # Check that the labels and outputs have the same numbers of recordings.
    assert(len(tmp_labels)==len(tmp_binary_outputs)==len(tmp_scalar_outputs))
    num_recordings = len(tmp_labels)

    # Rearrange the columns of the labels and the outputs to be consistent with the order of the classes.
    labels = np.zeros((num_recordings, num_classes), dtype=np.bool)
    for k, dx in enumerate(label_classes):
        j = classes.index(dx)
        labels[:, j] = tmp_labels[:, k]

    binary_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    scalar_outputs = np.zeros((num_recordings, num_classes), dtype=np.float64)
    for k, dx in enumerate(output_classes):
        j = classes.index(dx)
        binary_outputs[:, j] = tmp_binary_outputs[:, k]
        scalar_outputs[:, j] = tmp_scalar_outputs[:, k]

    return classes, labels, binary_outputs, scalar_outputs

# Load a table with row and column names.
def load_table(table_file):
    # The table should have the following form:
    #
    # ,    a,   b,   c
    # a, 1.2, 2.3, 3.4
    # b, 4.5, 5.6, 6.7
    # c, 7.8, 8.9, 9.0
    #
    table = list()
    with open(table_file, 'r') as f:
        for i, l in enumerate(f):
            arrs = [arr.strip() for arr in l.split(',')]
            table.append(arrs)

    # Define the numbers of rows and columns and check for errors.
    num_rows = len(table)-1
    if num_rows<1:
        raise Exception('The table {} is empty.'.format(table_file))

    num_cols = set(len(table[i])-1 for i in range(num_rows))
    if len(num_cols)!=1:
        raise Exception('The table {} has rows with different lengths.'.format(table_file))
    num_cols = min(num_cols)
    if num_cols<1:
        raise Exception('The table {} is empty.'.format(table_file))

    # Find the row and column labels.
    rows = [table[0][j+1] for j in range(num_rows)]
    cols = [table[i+1][0] for i in range(num_cols)]

    # Find the entries of the table.
    values = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            value = table[i+1][j+1]
            if is_number(value):
                values[i, j] = float(value)
            else:
                values[i, j] = float('nan')

    return rows, cols, values

# Load weights.
def load_weights(weight_file, classes):
    # Load the weight matrix.
    rows, cols, values = load_table(weight_file)
    assert(rows == cols)
    num_rows = len(rows)

    # Assign the entries of the weight matrix with rows and columns corresponding to the classes.
    num_classes = len(classes)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i, a in enumerate(rows):
        if a in classes:
            k = classes.index(a)
            for j, b in enumerate(rows):
                if b in classes:
                    l = classes.index(b)
                    weights[k, l] = values[i, j]
    np.fill_diagonal(weights, 1)

    return weights

# Compute recording-wise accuracy.
def compute_accuracy(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    num_correct_recordings = 0
    for i in range(num_recordings):
        if np.all(labels[i, :]==outputs[i, :]):
            num_correct_recordings += 1

    return float(num_correct_recordings) / float(num_recordings)

# Compute confusion matrices.
def compute_confusion_matrices(labels, outputs, normalize=False):
    # Compute a binary confusion matrix for each class k:
    #
    #     [TN_k FN_k]
    #     [FP_k TP_k]
    #
    # If the normalize variable is set to true, then normalize the contributions
    # to the confusion matrix by the number of labels per recording.
    num_recordings, num_classes = np.shape(labels)

    if not normalize:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')
    else:
        A = np.zeros((num_classes, 2, 2))
        for i in range(num_recordings):
            normalization = float(np.sum(labels[i, :]))
            for j in range(num_classes):
                if labels[i, j]==1 and outputs[i, j]==1: # TP
                    A[j, 1, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==1: # FP
                    A[j, 1, 0] += 1.0/normalization
                elif labels[i, j]==1 and outputs[i, j]==0: # FN
                    A[j, 0, 1] += 1.0/normalization
                elif labels[i, j]==0 and outputs[i, j]==0: # TN
                    A[j, 0, 0] += 1.0/normalization
                else: # This condition should not happen.
                    raise ValueError('Error in computing the confusion matrix.')

    return A

# Compute macro F-measure.
def compute_f_measure(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs)

    f_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if 2 * tp + fp + fn:
            f_measure[k] = float(2 * tp) / float(2 * tp + fp + fn)
        else:
            f_measure[k] = float('nan')

    macro_f_measure = np.nanmean(f_measure)

    return macro_f_measure

# Compute F-beta and G-beta measures from the unofficial phase of the Challenge.
def compute_beta_measures(labels, outputs, beta):
    num_recordings, num_classes = np.shape(labels)

    A = compute_confusion_matrices(labels, outputs, normalize=True)

    f_beta_measure = np.zeros(num_classes)
    g_beta_measure = np.zeros(num_classes)
    for k in range(num_classes):
        tp, fp, fn, tn = A[k, 1, 1], A[k, 1, 0], A[k, 0, 1], A[k, 0, 0]
        if (1+beta**2)*tp + fp + beta**2*fn:
            f_beta_measure[k] = float((1+beta**2)*tp) / float((1+beta**2)*tp + fp + beta**2*fn)
        else:
            f_beta_measure[k] = float('nan')
        if tp + fp + beta*fn:
            g_beta_measure[k] = float(tp) / float(tp + fp + beta*fn)
        else:
            g_beta_measure[k] = float('nan')

    macro_f_beta_measure = np.nanmean(f_beta_measure)
    macro_g_beta_measure = np.nanmean(g_beta_measure)

    return macro_f_beta_measure, macro_g_beta_measure

# Compute macro AUROC and macro AUPRC.
def compute_auc(labels, outputs):
    num_recordings, num_classes = np.shape(labels)

    # Compute and summarize the confusion matrices for each class across at distinct output values.
    auroc = np.zeros(num_classes)
    auprc = np.zeros(num_classes)

    for k in range(num_classes):
        # We only need to compute TPs, FPs, FNs, and TNs at distinct output values.
        thresholds = np.unique(outputs[:, k])
        thresholds = np.append(thresholds, thresholds[-1]+1)
        thresholds = thresholds[::-1]
        num_thresholds = len(thresholds)

        # Initialize the TPs, FPs, FNs, and TNs.
        tp = np.zeros(num_thresholds)
        fp = np.zeros(num_thresholds)
        fn = np.zeros(num_thresholds)
        tn = np.zeros(num_thresholds)
        fn[0] = np.sum(labels[:, k]==1)
        tn[0] = np.sum(labels[:, k]==0)

        # Find the indices that result in sorted output values.
        idx = np.argsort(outputs[:, k])[::-1]

        # Compute the TPs, FPs, FNs, and TNs for class k across thresholds.
        i = 0
        for j in range(1, num_thresholds):
            # Initialize TPs, FPs, FNs, and TNs using values at previous threshold.
            tp[j] = tp[j-1]
            fp[j] = fp[j-1]
            fn[j] = fn[j-1]
            tn[j] = tn[j-1]

            # Update the TPs, FPs, FNs, and TNs at i-th output value.
            while i < num_recordings and outputs[idx[i], k] >= thresholds[j]:
                if labels[idx[i], k]:
                    tp[j] += 1
                    fn[j] -= 1
                else:
                    fp[j] += 1
                    tn[j] -= 1
                i += 1

        # Summarize the TPs, FPs, FNs, and TNs for class k.
        tpr = np.zeros(num_thresholds)
        tnr = np.zeros(num_thresholds)
        ppv = np.zeros(num_thresholds)
        npv = np.zeros(num_thresholds)

        for j in range(num_thresholds):
            if tp[j] + fn[j]:
                tpr[j] = float(tp[j]) / float(tp[j] + fn[j])
            else:
                tpr[j] = float('nan')
            if fp[j] + tn[j]:
                tnr[j] = float(tn[j]) / float(fp[j] + tn[j])
            else:
                tnr[j] = float('nan')
            if tp[j] + fp[j]:
                ppv[j] = float(tp[j]) / float(tp[j] + fp[j])
            else:
                ppv[j] = float('nan')

        # Compute AUROC as the area under a piecewise linear function with TPR/
        # sensitivity (x-axis) and TNR/specificity (y-axis) and AUPRC as the area
        # under a piecewise constant with TPR/recall (x-axis) and PPV/precision
        # (y-axis) for class k.
        for j in range(num_thresholds-1):
            auroc[k] += 0.5 * (tpr[j+1] - tpr[j]) * (tnr[j+1] + tnr[j])
            auprc[k] += (tpr[j+1] - tpr[j]) * ppv[j+1]

    # Compute macro AUROC and macro AUPRC across classes.
    macro_auroc = np.nanmean(auroc)
    macro_auprc = np.nanmean(auprc)

    return macro_auroc, macro_auprc

# Compute modified confusion matrix for multi-class, multi-label tasks.
def compute_modified_confusion_matrix(labels, outputs):
    # Compute a binary multi-class, multi-label confusion matrix, where the rows
    # are the labels and the columns are the outputs.
    num_recordings, num_classes = np.shape(labels)
    A = np.zeros((num_classes, num_classes))

    # Iterate over all of the recordings.
    for i in range(num_recordings):
        num_outputs = float(np.sum(outputs[i, :]))
        # Iterate over all of the classes.
        for j in range(num_classes):
            # Assign full and/or partial credit for each positive class.
            if labels[i, j]:
                for k in range(num_classes):
                    if outputs[i, k]:
                        A[j, k] += 1.0/num_outputs

    return A

# Compute the evaluation metric for the Challenge.
def compute_challenge_metric(weights, labels, outputs, classes, normal):
    num_recordings, num_classes = np.shape(labels)
    normal_index = classes.index(normal)

    # Compute observed score.
    A = compute_modified_confusion_matrix(labels, outputs)
    observed_score = np.nansum(weights * A)

    # Compute score for model that always chooses the correct label(s).
    correct_outputs = labels
    A = compute_modified_confusion_matrix(labels, correct_outputs)
    correct_score = np.nansum(weights * A)

    # Compute score for model that always chooses the normal label.
    inactive_outputs = np.zeros((num_recordings, num_classes), dtype=np.bool)
    inactive_outputs[:, normal_index] = 1
    A = compute_modified_confusion_matrix(labels, inactive_outputs)
    inactive_score = np.nansum(weights * A)

    if correct_score != inactive_score:
        normalized_score = float(observed_score - inactive_score) / float(correct_score - inactive_score)
    else:
        normalized_score = float('nan')

    return normalized_score

if __name__ == '__main__':
    auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric = evaluate_12ECG_score(sys.argv[1], sys.argv[2])

    output_string = 'AUROC,AUPRC,Accuracy,F-measure,Fbeta-measure,Gbeta-measure,Challenge metric\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format(auroc, auprc, accuracy, f_measure, f_beta_measure, g_beta_measure, challenge_metric)
    if len(sys.argv) > 3:
        with open(sys.argv[3], 'w') as f:
            f.write(output_string)
    else:
        print(output_string)
