function p = predict(Theta1, Theta2, Theta3, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta3, 1);

p = zeros(size(X, 1), 1);

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
y3 = z_3';

% y2 is the output of the neural network (in this case we have only two
% layers, namely one hidden and one output layer)
if num_labels == 1 % One class
    p = sign(y3);
    p = bsxfun(@plus,p,1)./2;
else               % Multi class
    [dummy, p] = max(y3, [], 2);
    %[dummy, p] = max(y2, [], 1);
end
end
