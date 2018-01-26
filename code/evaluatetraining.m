function accuracy = evaluatetraining(nn_params, options)

X = options.data;
y = options.labels;
first_hidden_layer_size = options.first_hidden_size;
second_hidden_layer_size = options.second_hidden_size;
input_layer_size = options.input_size;
classes = options.output_size;

Theta1 = reshape(nn_params(1:first_hidden_layer_size * (input_layer_size + 1)), ...
                 first_hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (first_hidden_layer_size * (input_layer_size + 1))):...
                (first_hidden_layer_size * (input_layer_size + 1) + second_hidden_layer_size * (first_hidden_layer_size + 1))), ...
                 second_hidden_layer_size, (first_hidden_layer_size + 1));

Theta3 = reshape(nn_params((1 + (first_hidden_layer_size * (input_layer_size + 1) + second_hidden_layer_size * (first_hidden_layer_size + 1))):end), ...
                 classes, (second_hidden_layer_size + 1));

pred = predict(Theta1, Theta2, Theta3, X);

accuracy = mean(double(pred == y)) * 100;

