function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% z variable (input of sigmoid)
z = X*theta;

% Computes gradient of J(theta)
grad = (1/m)*(sigmoid(z)-y)'*X;
grad = grad';   % grad should have the same dimensions as theta

% Computes J(theta)
J = -(1/m)*(y'*log(sigmoid(z)) + (1-y)'*log(1-sigmoid(z)));

end
