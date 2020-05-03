function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% C and sigma for evaluating different models on cross-validation set
CVec = [1, 3, 10, 30, 100, 300, 1000, 3000];
sigmaVec = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% find best combination of C and sigma (which produces the least 
% classification error)
best_i = 1;
best_j = 1;
leastError = Inf;

for i=1:length(CVec)
    for j=1:length(sigmaVec)
        C = CVec(i);
        sigma = sigmaVec(j);
        
        % trains SVM with C and sigma
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        modelError = mean(double(predictions ~= yval));
        
        if (modelError < leastError)
            leastError = modelError;
            best_i = i;
            best_j = j;
        end;
    end;
end;

C = CVec(best_i);
sigma = sigmaVec(best_j);
end
