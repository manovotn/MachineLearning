function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
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

% NOTE this is LINEAR regresision, not logistic

%hypothesis
hypothesis = X * theta;

%unregularized cost
J = (1/(2*m)) * sum((hypothesis .- y).^2);

% compute regularization; yields scalar number
% note that we only add theta(2:end,:) -> leave out theta_zero
reg_cost = lambda/(2 * m) * sum(theta(2:end,:).^2);

% compute actual cost J
J = J + reg_cost;

% compute grad the same way we did w/o regularization
grad = 1/m .* X' * (hypothesis .- y);

% now add the regularization magic to all thetas except the first one
grad(2:end,:) = grad(2:end,:) + lambda/m .* theta(2:end,:);

% =========================================================================

grad = grad(:);

end
