function [bestEpsilon, bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

% intializes variables
bestEpsilon = 0;
bestF1 = 0;

% Computes the F1 score when choosing epsilon as the probability threshold 
% for classifying a value as anomaly. Stores the epsilon which yelds the 
% greater F1.
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % if p(xi) < epsilon, then xi is considered anomaly
    cvPredictions = (pval < epsilon);

    % metrics
    fp = sum((cvPredictions == 1)&(yval == 0));
    tp = sum((cvPredictions == 1)&(yval == 1));
    fn = sum((cvPredictions == 0)&(yval == 1));
    
    % performance measurements
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
  
    % F1 score
    F1 = 2*precision*recall/(precision+recall);

    % updates best F1
    if (F1>bestF1)
        bestF1 = F1;
        bestEpsilon = epsilon;
    end;
end;

end
