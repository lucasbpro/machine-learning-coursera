function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% constants
m = size(X,1);

% covariance matrix
Sigma = (1/m)*X'*X;

% SVD computes the eigenvectors and eigenvalues of the covariance matrix. 
[U, S, V] = svd(Sigma);

end
