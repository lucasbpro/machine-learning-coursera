function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z (where z can be a matrix, 
%       vector or scalar).

% Computes the sigmoid of each value of z 
g = 1./(1+exp(-z));

end
