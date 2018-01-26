function nn_params = FBGDmomentum(costFunction,initial_nn_params,options)

epochs = options.epochs;
learning_rate = options.learning_rate;
momentum = options.momentum;

nn_params = initial_nn_params;
delta = zeros(size(nn_params));
J_plot = zeros(1, epochs);
accuracy_plot = zeros(1, epochs);

for iter = 1:epochs
    [J, grad] = costFunction(nn_params);
    J_plot(iter) = J;
    accuracy = evaluatetraining(nn_params,options);
    accuracy_plot(iter) = accuracy;
    if mod(iter,5) == 0
        fprintf('Iter: %d Objective: %.4f\n',iter,J);
    end
    if mod(iter,20) == 0 
        %fprintf('\nTraining accuracy: %.4f\n\n',evaluatetraining(nn_params,options));
        fprintf('\nTraining accuracy: %.4f\n\n',accuracy);
    end
    
    delta = momentum*delta - learning_rate*grad;
    nn_params = nn_params + delta;
end

%% Plot and Store the Objective Function over Iterations
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<
t = 1:epochs;
figure;
comet(t, J_plot);

%% Plot and Store the Training Accuracy over Iterations
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<
t = 1:epochs;
figure;
comet(t, accuracy_plot);

