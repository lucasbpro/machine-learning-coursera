function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

m = size(X,1);         % size of training set
mu = mean(X);          % mean of features
sigma = std(X);        % std of features

% feature normalization
X_norm = (X - repmat(mu,[m,1]))./repmat(sigma,[m,1]);

end
