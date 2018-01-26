%% Multiclass PU Neural Network

clear all; close all; clc;
addpath(genpath(pwd));

%% Initialization
clear ; close all; clc

rng default;


%% =========== Part 0: Setup Parameters/Hyperparameters =============

% Setup the size of dataset
percent = 5;                            % Ratio between labeled/unlabeled data (percentage)       
num_labels = 200;                       % Total number of labels
num_unlabeled = 100*num_labels/percent; % Total number of data (labeled+unlabeled)

% Setup the hyperparameters of the model
input_layer_size  = 784;            % 28x28 Input Images of Digits
first_hidden_layer_size = 500;      % 100 hidden units
second_hidden_layer_size = 200;      % 50 hidden units        
classes = 10;                       % 10 classes/output units

lambda = 0.1;

% Setup the parameters of the learning algorithm (Gradient Descent with Momentum)
learning_rate = 0.0001;
momentum = 0.9;
epochs = 5000;


%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
X = loadMNISTImages('train-images-idx3-ubyte')';
y = loadMNISTLabels('train-labels-idx1-ubyte');
X = X(y<=classes,:);
y = y(y<=classes);

% Compute the positive class priors
m = size(X, 1);
total_pos = zeros(1,classes);
for i = 1:classes
    total_pos(i) = sum(y == i);
end
prior = total_pos/m; % Positive class prior for each dataset

% Create the Labeled and Unlabeled Sets
id = 1:num_unlabeled;
X = X(id,:);
y = y(id);
m = size(X, 1);
labels = zeros(m,classes);
for i = 1:num_labels
    labels(i,y(i)) = 1;
end


% % Train an autoencoder
% hiddenSize1 = 200;
% autoenc1 = trainAutoencoder(X',hiddenSize1, ...
%     'MaxEpochs',1000, ...
%     'L2WeightRegularization',0.004, ...
%     'SparsityRegularization',4, ...
%     'SparsityProportion',0.15, ...
%     'ScaleData', false);
% 
% feat1 = encode(autoenc1, X');
% X = feat1';
% input_layer_size = size(feat1, 1);
%end of code for autoencoder

% % Random display of 100 samples
% sel = randperm(m);
% sel = sel(1:100);
% 
% displayData(X(sel, :));
% 
% fprintf('Program paused. Press enter to continue.\n');
% pause;


%% ================ Part 2: Initializing Pameters ================

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, first_hidden_layer_size);
initial_Theta2 = randInitializeWeights(first_hidden_layer_size, second_hidden_layer_size);
initial_Theta3 = randInitializeWeights(second_hidden_layer_size, classes);


% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];


% %% =============== Part 3: Gradient Checking ===============
% 
% fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')
% 
% %  Check gradients by running checkNNGradients
% checkNNGradients(lambda);
% 
% % Also output the costFunction debugging values
% debug_J  = nnCostFunction(initial_nn_params, input_layer_size, ...
%                           hidden_layer_size, classes, X, labels, ...
%                           prior, lambda);
% 
% fprintf('Program paused. Press enter to quit.\n');
% pause; return;


%% =================== Part 4: Training NN ===================

fprintf('\nTraining Neural Network... \n')

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   first_hidden_layer_size, ...
                                   second_hidden_layer_size, ...
                                   classes, X, labels, prior, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
options = [];
options.epochs = epochs;
options.learning_rate = learning_rate;
options.momentum = momentum;
options.data = X;                           % EVALUATION
options.labels = y;                         % EVALUATION
options.first_hidden_size = first_hidden_layer_size;    % EVALUATION
options.second_hidden_size = second_hidden_layer_size
options.input_size = input_layer_size;      % EVALUATION
options.output_size = classes;              % EVALUATION
nn_params = FBGDmomentum(costFunction,initial_nn_params,options);

% Obtain Theta1 and Theta2 back from nn_params
             
Theta1 = reshape(nn_params(1:first_hidden_layer_size * (input_layer_size + 1)), ...
                 first_hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (first_hidden_layer_size * (input_layer_size + 1))):...
                (first_hidden_layer_size * (input_layer_size + 1) + second_hidden_layer_size * (first_hidden_layer_size + 1))), ...
                 second_hidden_layer_size, (first_hidden_layer_size + 1));

Theta3 = reshape(nn_params((1 + (first_hidden_layer_size * (input_layer_size + 1) + second_hidden_layer_size * (first_hidden_layer_size + 1))):end), ...
                 classes, (second_hidden_layer_size + 1));

% fprintf('Program paused. Press enter to continue.\n');
% pause;


%% ================= Part 5: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

% fprintf('\nVisualizing Weights Learnt in the First Layer... \n')
% 
% displayData(Theta1(:, 2:end));
% 
% fprintf('\nProgram paused. Press enter to continue.\n');
% pause;

%% ================= Part 6: Prediction =================

pred = predict(Theta1, Theta2, Theta3, X);

fprintf('\nVisualizing the Confusion Matrix...\n');

targets = zeros(classes,m);
outputs = zeros(classes,m);
for i = 1:m
    targets(y(i),i) = 1;
    outputs(pred(i),i) = 1;
end
figure;
plotconfusion(targets,outputs);

%% Save the Confusion Matrix
% >>>>>>>>>>>>>> ADD CODE HERE <<<<<<<<<<<<<<<<<

[C, order] = confusionmat(y', pred');

fprintf('Program paused. Press enter to continue.\n');
pause;

clear all; close all; clc;