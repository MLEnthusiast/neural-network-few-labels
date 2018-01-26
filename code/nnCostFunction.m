function [J,grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   first_hidden_layer_size, ...
                                   second_hidden_layer_size, ...
                                   classes, ...
                                   X, labels, prior, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, classes, ...
%   X, labels, prior, lambda) computes the cost and gradient of the 
%   neural network. The parameters for the neural network are "unrolled" 
%   into the vector nn_params and need to be converted back into the weight
%   matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:first_hidden_layer_size * (input_layer_size + 1)), ...
                 first_hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (first_hidden_layer_size * (input_layer_size + 1))):...
                (first_hidden_layer_size * (input_layer_size + 1) + second_hidden_layer_size * (first_hidden_layer_size + 1))), ...
                 second_hidden_layer_size, (first_hidden_layer_size + 1));

Theta3 = reshape(nn_params((1 + (first_hidden_layer_size * (input_layer_size + 1) + second_hidden_layer_size * (first_hidden_layer_size + 1))):end), ...
                 classes, (second_hidden_layer_size + 1));

m = size(X, 1);             % Number of total samples
K = size(labels,2);         % Number of classes/output neurons
unl = m-sum(sum(labels));   % Number of unlabeled data
pos = sum(labels);          % Size of each positive labeled dataset
pos = pos(:);
prior = prior(:);

         
%% Forward Propagation (Vectorized Form)
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<
y_0 = [ones(1, m); X'];
z_1 = Theta1 * y_0;
y_1 = z_1;
y_1(y_1 <= 0) = 0;
y_1 = [ones(1, m); y_1];
z_2 = Theta2 * y_1;
y_2 = z_2;
y_2(y_2 <= 0) = 0;
y_2 = [ones(1, m); y_2];
z_3 = Theta3 * y_2;
y_3 = z_3;

%% Backward Propagation (Vectorized Form)
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<
Y = labels';
lab_coeff = diag(-(prior./pos));
unlab_coeff = diag(1./(m-pos));
delta_3 = lab_coeff*(Y==1) + unlab_coeff*((y_3>1).*(Y==0)) + unlab_coeff*(0.5*(y_3>=-1).*(y_3<=1).*(Y==0));
delta_2 = (Theta3(:,2:end)'*delta_3).*(z_2>0);
delta_1 = (Theta2(:,2:end)'*delta_2).*(z_1>0);

%% Gradient Computation (Vectorized Form)
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<
%Theta1_grad = zeros(size(Theta1));
%Theta2_grad = zeros(size(Thet2a));

Theta1_grad = delta_1*y_0';
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + 2*lambda*Theta1(:,2:end);
Theta2_grad = delta_2*y_1';
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + 2*lambda*Theta2(:,2:end);
Theta3_grad = delta_3 * y_2';
Theta3_grad(:,2:end) = Theta3_grad(:,2:end) + 2*lambda*Theta3(:,2:end);

%% Cost Computation (Vectorized Form)
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<
temp1 = lab_coeff*((y_3).*(Y==1));
temp2 = unlab_coeff*((y_3).*((y_3>1).*(Y==0)));
temp3 = unlab_coeff*(0.5 + (0.5*((y_3).*((y_3>=-1).*(y_3<=1).*(Y==0)))));
J = sum(sum(temp1)) + sum(sum(temp2)) + sum(sum(temp3)) + ...
    lambda*(sum(sum(Theta1_grad(:,2:end).^2))+ ...    % regularizer term
    sum(sum(Theta2_grad(:,2:end).^2))+...             % regularizer term
    sum(sum(Theta3_grad(:,2:end).^2)));               % regularizer term


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:) ; Theta3_grad(:)];

end
