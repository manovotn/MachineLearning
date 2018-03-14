function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % PREVIOUS SOLUTION - WORKS BUT NOT FULLY VECTORIZED
    %new_theta = zeros(size(theta));
    %for i = 1:length(theta)
     % delta = (1/m) .* sum((sum((theta' .* X), 2) .- y) .* X(:,i));
      %disp('this is delta');
      %disp(delta);
      %disp('old theta')
      %disp(theta(i,1))
      %new_theta(i,1) = theta(i,1) .- (alpha .* delta);
      %disp('new theta')
      %disp(new_theta(i,1))
    %end
    %theta = new_theta;
    
    %VECTORIZED SOLUTION
    % compute hypothesis
    hypothesis = X * theta;
    % compute all errors
    errors = hypothesis .- y;
    % compute decrement vector for all thetas
    decrement = alpha * (1/m) * errors' * X;
    % simultaneous update on all thetas
    theta = theta - decrement';


    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    % sanity check - throw error if previous cost (J) was lower then the one we got now
    if (iter > 1 && (J_history(iter -1) < J_history(iter)))
      error('previous iteration had lower J then current one!');
    end

end

end
