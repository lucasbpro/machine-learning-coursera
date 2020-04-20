function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% Theta1 is hidden_layer_size x (input_layer_size+1)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% Theta2 is output_layer_size (=num_labels) x (hidden_layer_size+1)
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
% Setup some useful variables
m = size(X, 1);   

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
%% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% adds column of ones to include bias term in computation
A1 = [ones(m,1) X]; 
A2 = sigmoid(A1*Theta1');      % A2 is m x hidden_layer_size

% adds column of ones to include bias term in computation
A2 = [ones(m,1) A2];           % A2 is now m x (hidden_layer_size+1)
A3 = sigmoid(A2*Theta2');      % A3 is m x num_labels

% calculates cost-function J for NN(Theta1,Theta2)
J = 0; 
for i=1:m
    
    % h_xi is the output of the NN for training example xi
    h_xi = A3(i,:)';
    
    % yk is a vector with value 1 at the position which corresponds to the
    % actual label of x_i and zeros at other positions
    yk = zeros(num_labels,1);
    yk(y(i)) = 1;
    
    % calculates contribution for a single training example xi
    J = J -(1/m)*( yk'*log(h_xi) + (1-yk)'*log(1-h_xi) );
end;

% Extract the NN parameters, excluding the bias parameters
Theta1_excluding_bias = Theta1(:,2:end);
Theta2_excluding_bias = Theta2(:,2:end);
nn_params_excluding_bias = [Theta1_excluding_bias(:); Theta2_excluding_bias(:)];   

% Adds regularization term to cost-function J
J = J + (lambda/(2*m))*sum(nn_params_excluding_bias.^2); 

%% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% initialize accumulated gradient matrices
D1 = zeros(hidden_layer_size,input_layer_size+1);
D2 = zeros(num_labels,hidden_layer_size+1);

% for each training example
for i=1:m
    
    % forward pass
    a1 = X(i,:)';
    a1 = [1; a1];
    z2 = Theta1*a1;
    a2 = sigmoid(z2);  
    a2 = [1; a2];
    z3 = Theta2*a2;
    a3 = sigmoid(z3);
        
    % reset yk variable
    yk = zeros(num_labels,1);
    
    % yk is a vector with value 1 at the position which corresponds to the
    % actual label of x_i and zeros at other positions
    yk(y(i)) = 1;
    
    % output error
    delta_3 = a3 - yk;
    
    % hidden layer error   
    delta_2 = Theta2'*delta_3.*(a2.*(1-a2));
    
    % accumulated gradient matrices
    D2 = D2 + delta_3*a2';
    D1 = D1 + delta_2(2:end)*a1';
end;

% compute gradients
Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

%% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = Theta1_grad + (lambda/m)*[zeros(hidden_layer_size,1) Theta1(:,2:end)];
Theta2_grad = Theta2_grad + (lambda/m)*[zeros(num_labels,1) Theta2(:,2:end)];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
