function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);          % number of training examples

% history of cost function throughout gradient descent alogrithm
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % single gradient step on the parameter vector theta
    S = (X*theta - y)'*X;    % this is a summatory (refer to Grad. Desc. Algorithm)
    theta = theta - (1/m)*alpha*S';

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
