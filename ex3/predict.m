function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% training set size
m = size(X, 1);

% number of classes (labels)
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(m,1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% adds column of ones to include bias term in computation
A1 = [ones(m,1) X]; 
A2 = sigmoid(A1*Theta1');      % A2 is m x 25, Theta1' is 401 x 25

% adds column of ones to include bias term in computation
A2 = [ones(m,1) A2];           % A2 is now m x 26 
A3 = sigmoid(A2*Theta2');      % A3 is m x 10, Theta2' is 26 x 10 

[~, p] = max(A3, [], 2);       % p is the vector containing the predicted 
                               % class for training examples

% =========================================================================

end
