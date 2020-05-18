function [mu, sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

m = size(X,1);

% mean
mu = mean(X);

% variance
sigma2 = mean((X-repmat(mu,m,1)).^2);

mu = mu';
sigma2 = sigma2';

end
