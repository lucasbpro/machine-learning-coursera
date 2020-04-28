function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

%  number of training examples
m = length(y); 

% predicted y using paramaters theta
h_theta = X*theta;

% Computes J(theta) without regularization term
J = (1/(2*m))*sum((h_theta - y).^2) + (lambda/(2*m))*sum([0; theta(2:end)].^2);

% Computes gradient of J(theta) including regularization term
grad = (1/m)*((h_theta - y)'*X) + (lambda/m)*[0; theta(2:end)]';
grad = grad(:);

end
