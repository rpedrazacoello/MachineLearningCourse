function [J grad] = nnCostFunction(nn_params, ...
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
m = size(X, 1); % m = 5000
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));




% Tamaño de matrices que se utilizan

% size (X) = (5000 x 400)

% size (y) = (5000 x 1)

% size(Theta1) = (25 x 401)

% size(Theta2) = (10 x 26)




% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a1 = [ones(m, 1) X];
% size (a1) = (5000 x 401)

a2 = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) a2];
% size (a2) = (5000 x 26)

a3 = sigmoid(a2 * Theta2'); %This is the output layer, h(x)
% size(a3) = (5000 x 10)

%Esto lo que hace es hacer una matriz que en cada fila tendra el valor 1, 2, 3,.., 10 (num_labels=10)
temp = repmat([1:num_labels], m, 1); % (5000 x 10)

%Esto lo que hace es hacer una matriz donde se repetira el valor que tiene y para cada X, ejemplo una fila podria ser 10 10 10 10 10 10 10 10 10 10
temp2 = repmat(y, 1, num_labels); % (5000 x 10)

%Lo que hace es comparar valor por valor temp y temp2, si el valor es igual entonces pondr un 1, si no, pondra un 0
y = temp == temp2; % (5000 x 10)

tempJ1 = -y .* log(a3);
tempJ0 = (1 - y) .* log(1 - a3);

Jtemp = 1/m * sum(sum(tempJ1 - tempJ0));

regTheta1 = Theta1(:, 2:end);
regTheta2 = Theta2(:, 2:end);

RegTemp = (lambda/(2*m)) * (sum(sum(regTheta1 .^2)) + sum(sum(regTheta2 .^2)));

J = Jtemp + RegTemp;

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

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for t = 1:m,
	a1t = a1(t, :);
	a2t = a2(t, :);
	a3t = a3(t, :);
	yt = y(t, :);

	error3 = (a3t - yt);
	z = [1; Theta1 * a1t']; % size = (26 x 1)
	error2 = Theta2' * error3' .* sigmoidGradient(z);

	delta1 = delta1 + (error2(2:end) * a1t);
	delta2 = delta2 + (error3' * a2t);
end;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

tempTheta1 = Theta1(:, 2:end); % You take out the first column of Theta because that's not supposed to be regularized
tempTheta2 = Theta2(:, 2:end);

tempTheta1 = [zeros(size(tempTheta1, 1), 1) tempTheta1]; % You add a first column of zeros to Theta
tempTheta2 = [zeros(size(tempTheta2, 1), 1) tempTheta2];

Theta1_grad = ((1/m) * delta1) + ((lambda/m) * tempTheta1);
Theta2_grad = ((1/m) * delta2) + ((lambda/m) * tempTheta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
