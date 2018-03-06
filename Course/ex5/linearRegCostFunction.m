Jfunction [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% size(theta) = (2 x 1) 

% size(X) = (12 x 2)

% size (y) = (12 x 1)

regTheta = theta;
regTheta(1) = 0;

hx = X * theta;

Jtemp = sum((hx - y).^2);
Jreg = lambda/(2 * m) * sum(regTheta.^2);
J = (1 /(2*m) * Jtemp) + Jreg;

error = hx - y; 
% size (error) = (12 x 1)
grad = (1/m) * (X' * error) + ((lambda/m) * regTheta);

% size (grad) = (12 x 2)










% =========================================================================

% grad = grad(:);

end
