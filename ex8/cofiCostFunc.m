function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

% Notes: X - num_movies  x num_features    --->  matrix of movie features
%        Theta - num_users  x num_features --->  matrix of user features
%        Y - num_movies x num_users        --->  matrix of user ratings of movies
%        R - num_movies x num_users        --->  R(i, j) = 1 if the i-th movie was rated by the j-th user


% Cost function for collaborative filtering
J = 0.5*sum(sum((R.*(X*Theta' - Y)).^2)) + ...
    + (lambda/2)*sum(sum(Theta.^2)) + ...
    + (lambda/2)*sum(sum(X.^2));

% X_grad - num_movies x num_features matrix, containing the 
%          partial derivatives w.r.t. to each element of X
X_grad = (R.*(X*Theta' - Y))*Theta + lambda*X;

% Theta_grad - num_users x num_features matrix, containing the 
%              partial derivatives w.r.t. to each element of Theta
Theta_grad = (R.*(X*Theta' - Y))'*X + lambda*Theta;

% Unroll
grad = [X_grad(:); Theta_grad(:)];

end
