% This file contains functions for evaluating algorithms for the 2020 PhysioNet/
% CinC Challenge. You can run it as follows:
%
%   evaluate_12ECG_score(labels, outputs, 'scores.csv')
%
% where 'labels' is a directory containing files with labels, 'outputs' is a
% directory containing files with outputs, and 'scores.csv' (optional) is a
% collection of scores for the outputs.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The evaluate_scores function computes a Fbeta measure and a generalizatoin of
% the Jaccard measure but giving missed diagnosis twice as much weight as
% correct diagnoses and false alarms
%
% Inputs:
%   'label_directory' is the input directory with the headers and #dx with the true labels
%
%   'output_directory' is a directory of comma-delimited text files, where
%   the first row of the file is the predictive labels for each class and
%   the second row of the file is the probability of the class label. 
%   Note that there must be a output for every label.
%
% Outputs:
%
%   'f_beta' is F_beta measure, with beta = 2
%
%   'g_beta' is a generalization of the Jaccard measures but giving missed 
%   diagnoses twice as much weight as correct diagnoses and false alarms, beta = 2
%
%    'accuracy' is accuracy
%
%    'f_measure' is F-measure
%
% Example:
%   Omitted due to length. See the below examples.


function evaluate_12ECG_score(label_directory, output_directory, output_file)

    % Set parameters.
    label_header       = '12ECGLabel';
    output_header  = 'OutputLabel';
    probability_header = 'OutputProbability';

    beta = 2;

    % Find labels and output
    label_files={};

    for f = dir(label_directory)'
        if exist(fullfile(label_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'hea')
            label_files{end + 1} = f.name;
        end
    end

    output_files={};
    for f = dir(output_directory)'
        if exist(fullfile(output_directory, f.name), 'file') == 2 && f.name(1) ~= '.' && all(f.name(end - 2 : end) == 'csv')
            output_files{end + 1} = f.name;
        end
    end

    if length(label_files) ~= length(output_files)
        error('Numbers of label and output files must be the same.');
    end


    % Load labels and outputs.
    num_files = length(label_files);
    labels=[];
    output=[];
    output_probabilities=[];

    classes = get_classes(label_directory,label_files);


    for k =1:num_files

	[recording_label,classes_label,single_recording_labels]=get_true_labels([label_directory filesep label_files{k}],classes);

        fid2=fopen(fullfile(output_directory,output_files{k}));
        recording_output = fgetl(fid2);
        classes_output = strsplit(fgetl(fid2),',');
        single_recording_output = cellfun(@str2double,strsplit(fgetl(fid2),','));
	single_probabilities_output=cellfun(@str2double,strsplit(fgetl(fid2),','));
        fclose(fid2);

	% Check labels and outputs for errors.
	if ~(strcmp(classes_label,classes_output))
		error('Numbers of labels and outputs for a file must be the same.');
	end

	if ~(length(single_recording_labels) == length(single_recording_output) || length(single_recording_output) == length(single_probabilities_output))
        	error('Numbers of labels and outputs for a file must be the same.');
	end

	labels = [labels ; single_recording_labels];
	output = [output ; single_recording_output];
	output_probabilities = [output_probabilities ; single_probabilities_output];

    end

    num_classes = length(classes_label);


    % Compute F_beta measure and the generalization of the Jaccard index
    [accuracy,f_measure,f_beta,g_beta] = compute_beta_score(labels, output, beta, num_classes);


    % Compute AUC, accuracy, and F-measure.

    [auroc,auprc] = compute_auc(labels, output_probabilities,num_classes);


    % Output results.
    output_string = sprintf('AUROC|AUPRC|Accuracy|F-measure|Fbeta-measure|Gbeta-measure\n%.3f|%.3f|%.3f|%.3f|%.3f|%.3f',...
                             auroc, auprc, accuracy, f_measure, f_beta,g_beta);
                        
    switch nargin
        case 2
            disp(output_string)
        case 3
            fid = fopen(output_file, 'wt');
            fprintf(fid, output_string);
            fclose(fid);
    end


end



% The compute_beta_score function computes the Fbeta-measure giving an specific beta value
% and the G value define at the begining of the file
%
% Inputs:
%   'labels' are the true classes of the recording
%
%   'outputs' are the output classes of your model
%
%   'beta' is the weight
%
% Output:
%   f_beta, Fbeta measure given an specific beta
%
%   g_beta, generalization of the Jaccard measure with a beta weigth

function [accuracy,f_measure,f_beta,g_beta] = compute_beta_score(labels, outputs,beta,num_classes)
    % Check inputs for errors.
    if length(outputs) ~= length(labels)
        error('Numbers of outputs and labels must be the same.');
    end

    [num_recordings,num_classes_from_lab] = size(labels);

        % Check inputs for errors.
    if length(num_classes) ~= length(num_classes_from_lab)
        error('Numbers of classes and labels must be the same.');
    end

    % Populate contingency table.

    fbeta_l = zeros(1,num_classes);
    gbeta_l = zeros(1,num_classes);
    fmeasure_l = zeros(1,num_classes);
    accuracy_l = zeros(1,num_classes);

    f_beta = 0;
    g_beta = 0;
    f_measure = 0;
    accuracy = 0;

    % Weigth function
    C_l = ones(1,num_classes);

    for j=1:num_classes
	tp = 0;
	fp = 0;
	fn = 0;
	tn = 0;

	for i = 1 : num_recordings

		num_labels = sum(labels(i,:));

	        if labels(i,j)==1 && outputs(i,j)==1
	            tp = tp + 1/num_labels;
	        elseif labels(i,j)~=1 && outputs(i,j)==1
	            fp = fp + 1/num_labels;
	        elseif labels(i,j)==1 && outputs(i,j)~=1
    		    fn = fn + 1/num_labels;
	        elseif labels(i,j)~=1 && outputs(i,j)~=1
	            tn = tn + 1/num_labels;
	        end
	end

	% Summarize contingency table.
        if ((1+beta^2)*tp + (beta*fn) + fp) > 0
	        fbeta_l(j) = ((1+beta^2)*tp) / ((1+beta^2)*tp + (beta^2*fn) + fp);
        else
        	fbeta_l(j) = 1;
        end

	if (tp + (beta*fn) + fp) > 0
	        gbeta_l(j) = tp / (tp + (beta*fn) + fp);
	else
	        gbeta_l(j) = 1;
	end

	if (tp + fp + fn + tn) > 0
	        accuracy_l(j) = (tp+tn) / (tp+fp+fn+tn);
	else
	        accuracy_l(j) = 1;
	end

	if (2*tp + fp + tn) >0
		fmeasure_l(j) = (2*tp)/((2*tp)+fp+fn);
	else
		fmeasure_l(j) = 1;
	end

    end

    for i = 1:num_classes
	    f_beta = f_beta + fbeta_l(i)*C_l(i);
            g_beta = g_beta + gbeta_l(i)*C_l(i);
            f_measure = f_measure + fmeasure_l(i)*C_l(i);
            accuracy = accuracy + accuracy_l(i)*C_l(i);
    end

    f_beta = f_beta/num_classes;
    g_beta = g_beta/num_classes;
    f_measure = f_measure/num_classes;
    accuracy = accuracy/num_classes;

end

% The compute_auc function computes AUROC and AUPRC as well as other summary
% statistics (TP, FP, FN, TN, TPR, TNR, PPV, NPV, etc.) that can be exposed
% from this function.
%
% Inputs:
%   'labels' are the true classes of the recording
%
%   'output' are the output classes of your model
%
%   'beta' is the weight
%
%
% Outputs:
%   'auroc' is a scalar that gives the AUROC of the algorithm using its
%   output probabilities, where specificity is interpolated for intermediate
%   sensitivity values.
%
%   'auprc' is a scalar that gives the AUPRC of the algorithm using its
%   output probabilities, where precision is a piecewise constant function of
%   recall.
%

function [auroc, auprc] = compute_auc(labels,probabilities,num_classes)

    % Check inputs for errors.
    if length(probabilities) ~= length(labels)
        error('Numbers of probabilities and labels must be the same.');
    end

    auroc_l = zeros(1,num_classes);
    auprc_l = zeros(1,num_classes);

    auroc = 0;
    auprc = 0;

    % Weigth function 
    C_l = ones(1,num_classes);

    [num_recordings,num_classes_from_lab] = size(labels);

    for k = 1:num_classes
	    % Find probabilities thresholds.
	    thresholds = flipud(unique(probabilities(:,k)));

	    if thresholds(1) ~= 1
	        thresholds = [1; thresholds];
	    end

	    if thresholds(end) ~= 0
	        thresholds = [thresholds; 0];
	    end

	    m = length(thresholds);

	    % Populate contingency table across probabilities thresholds.
	    tp = zeros(1, m);
	    fp = zeros(1, m);
	    fn = zeros(1, m);
	    tn = zeros(1, m);

	    % Find indices that sort predicted probabilities from largest to smallest.
	    [~, idx] = sort(probabilities(:,k), 'descend');
	

	    i = 1;
	    for j = 1 : m
	        % Initialize contingency table for j-th probabilities threshold.
	        if j == 1
	            tp(j) = 0;
	            fp(j) = 0;
	            fn(j) = sum(labels(:,k));
	            tn(j) = num_recordings - fn(j);
	        else
	            tp(j) = tp(j - 1);
	            fp(j) = fp(j - 1);
	            fn(j) = fn(j - 1);
	            tn(j) = tn(j - 1);
	        end

		% Update contingency table for i-th largest probabilities probability.
	        while i <= num_recordings && probabilities(idx(i),k) >= thresholds(j)
	            if labels(idx(i),k) == 1
	                tp(j) = tp(j) + 1;
	                fn(j) = fn(j) - 1;
	            else
	                fp(j) = fp(j) + 1;
	                tn(j) = tn(j) - 1;
	            end
	            i = i + 1;
	        end
	    end

	    % Summarize contingency table.
	    tpr = zeros(1, m);
	    tnr = zeros(1, m);
	    ppv = zeros(1, m);
	    npv = zeros(1, m);

	    for j = 1 : m
	        if tp(j) + fn(j) > 0
	            tpr(j) = tp(j) / (tp(j) + fn(j));
	        else
	            tpr(j) = 1;
	        end

	        if fp(j) + tn(j) > 0
	            tnr(j) = tn(j) / (fp(j) + tn(j));
	        else
	            tnr(j) = 1;
	        end

	        if tp(j) + fp(j) > 0
	            ppv(j) = tp(j) / (tp(j) + fp(j));
	        else
	            ppv(j) = 1;
	        end

	        if fn(j) + tn(j) > 0
	            npv(j) = tn(j) / (fn(j) + tn(j));
	        else
	            npv(j) = 1;
	        end
	    end

	    % Compute AUROC as the area under a piecewise linear function of TPR /
	    % sensitivity (x-axis) and TNR / specificity (y-axis) and AUPRC as the area
	    % under a piecewise constant of TPR / recall (x-axis) and PPV / precision
	    % (y-axis).

	    for j = 1 : m - 1
	        auroc_l(k) = auroc_l(k) + 0.5 * (tpr(j + 1) - tpr(j)) * (tnr(j + 1) + tnr(j));
	        auprc_l(k) = auprc_l(k) + (tpr(j + 1) - tpr(j)) * ppv(j + 1);
	    end
    end

    for i =1:num_classes
	    auroc = auroc + auroc_l(i)*C_l(i);
	    auprc = auprc + auprc_l(i)*C_l(i);
    end

    auroc = auroc/num_classes;
    auprc = auprc/num_classes;
end

% function to obtain the true labels

function [recording_label,classes_label,single_recording_labels]=get_true_labels(input_file,classes)

	classes_label=classes;
	single_recording_labels=zeros(1,length(classes));

	fid=fopen(input_file);
        tline = fgetl(fid);
	tmp_str = strsplit(tline,' ');
	recording_label = tmp_str{1};

        tlines = cell(0,1);
        while ischar(tline)
	        tlines{end+1,1} = tline;
                tline = fgetl(fid);
        	if startsWith(tline,'#Dx')
                        tmp = strsplit(tline,': ');
                        tmp_c = strsplit(tmp{2},',');
                        for j=1:length(tmp_c)
                	        idx2 = find(strcmp(classes,tmp_c{j}));
				single_recording_labels(idx2)=1;
                        end
			break
                end
	end
end


% find unique number of classes
function classes = get_classes(input_directory,files)

        classes={};
        num_files = length(files);
        k=1;
        for i = 1:num_files
                input_file = fullfile(input_directory, files{i});
                fid=fopen(input_file);
                tline = fgetl(fid);
                tlines = cell(0,1);
		while ischar(tline)
                    tlines{end+1,1} = tline;
                    tline = fgetl(fid);
                        if startsWith(tline,'#Dx')
                                tmp = strsplit(tline,': ');
                                tmp_c = strsplit(tmp{2},',');
                                for j=1:length(tmp_c)
                                        idx2 = find(strcmp(classes,tmp_c{j}));
                                        if isempty(idx2)
                                                classes{k}=tmp_c{j};
                                                k=k+1;
                                        end
                                end
                        break
                        end
                end
                fclose(fid);
        end
        classes=sort(classes);
end

