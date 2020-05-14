function X_rec = recoverData(Z, U, K)
%RECOVERDATA Recovers an approximation of the original data when using the 
%projected data
%   X_rec = RECOVERDATA(Z, U, K) recovers an approximation the 
%   original data that has been reduced to K dimensions. It returns the
%   approximate reconstruction in X_rec.
%

% Computes the approximation of the data by projecting back onto the 
% original space using the top K eigenvectors in U.
X_rec = Z*U(:,1:K)';

% Note:
% For the i-th example Z(i,:), the (approximate)
% recovered data for dimension j is given as follows:
%       v = Z(i, :)';
%       recovered_j = v' * U(j, 1:K)';
