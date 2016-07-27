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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%%
%converting y

Y=zeros(size(y,1),num_labels);
for(i=1:size(y,1))
    tmp=zeros(num_labels,1);
    tmp(y(i),1)=1;
    Y(i,:)=tmp;
end

%%

%feedforward

X = [ones(m, 1) X];

z2=Theta1*X';

a2=sigmoid(z2);



a2= [ones(1, m); a2];

z3=Theta2*a2;

a3=sigmoid(z3);

%%
%Cost Function

theta1_withoutBiasSqaured=Theta1(1:end,2:end).^2;
theta2_withoutBiasSquared=Theta2(1:end,2:end).^2;

h_trans=a3';

cost=log(h_trans).*Y+log(1-h_trans).*(1-Y);

J=(-1/m)*sum(cost(:))+ (lambda/(2*m))*(sum(theta1_withoutBiasSqaured(:)) +sum(theta2_withoutBiasSquared(:)) );


%%
%Back propagation

error3=h_trans-Y;

error2=(error3*Theta2(:,2:end)).* sigmoidGradient(z2)';

delta1=error2'*X;

delta2=(a2*error3)';

Theta1(:,1)=0; %for regularaztion you do not need the bias
Theta2(:,1)=0; %for regularaztion you do not need the bias

Theta1_grad=(1/m)*delta1 +(lambda/m)*Theta1;

Theta2_grad=(1/m)*delta2+(lambda/m)*Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
