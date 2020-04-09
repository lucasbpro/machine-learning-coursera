function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% z variable (input of sigmoid)
z = X*theta;

% Computes gradient of J(theta) without regularization term
grad = (1/m)*(sigmoid(z)-y)'*X;
grad = grad';   % grad should have the same dimensions as theta

% adds regularization term
grad = grad + (lambda/m)*[0; theta(2:end)];

% Computes J(theta) without regularization term
J = -(1/m)*(y'*log(sigmoid(z)) + (1-y)'*log(1-sigmoid(z)));

% adds regularization term
J = J + (lambda/(2*m))*sum([0; theta(2:end)].^2);

end
