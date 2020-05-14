function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% Initialize Z
Z = zeros(size(X, 1), K);

% computes the projection of the data using only the top K eigenvectors in
% U (first K columns).
for k=1:K
    Z(:,k) = X*U(:, k);   % X(i,:) is a unit vector (norm = 1)
end;                 

end
