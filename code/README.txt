================= PROGRAMMING ASSIGNMENT ====================

--------------------- PHASE 1: Coding -----------------------

1. Add code in nnCostFunction.m starting from line 33.
   This file returns the objective function and the gradient
   computed at the current values of parameters (e.g. Theta1
   and Theta2) and is used by the optimization algorithm to 
   train the network.
   Remember to use a vectorized form to exploit parallel 
   computing in MATLAB and to speed up the training.

2. Add code in predict.m starting from line 12.
   This file predicts the labels for a given set of samples
   and is necessary to evaluate the performance of the network.
   Remember to use a vectorized form to exploit parallel 
   computing in MATLAB and to speed up the training.

-------------- PHASE 2: Checking implementation --------------

1. You are ready to check if your implemented gradient is
   compatible with your implemented objective function.
   To do this, uncomment lines 80 to 93 in ex.m and then run 
   the script.
   This step compares your implemented gradient against the 
   numerical gradient computed on your implemented cost 
   function. The distance between these two gradients must be
   less than 1e-9, otherwise your implementation is wrong.

2. Comment lines 80 to 93 in ex.m and run the script.
   With the following setup in ex.m,

   line 14:         % Setup the size of dataset
   line 15:         percent = 5;
   line 16:         num_labels = 200;
   line 17:         num_unlabeled = 100*num_labels/percent;
   line 18:
   line 19:         % Setup the hyperparameters of the model
   line 20:         input_layer_size  = 784;
   line 21:         hidden_layer_size = 100;
   line 22:         classes = 10;
   line 23:
   line 24:         lambda = 0.1;
   line 25:
   line 26:         % Setup the parameters of the ...
   line 27:         learning_rate = 0.0001;
   line 28:         momentum = 0.9;
   line 29:         epochs = 5000;

    You should achieve a training accuracy >= 70.6%. If you 
    achieve lower performance. Then your implementation is
    wrong.


-------------  PHASE 3: Improving Performance ---------------

Are you able to achieve 90% of training accuracy?

1. Add code in FBGDmomentum.m in order to plot and store the 
   learning curves, namely the objective and the training 
   accuracy over all iterations.

2. Add code in ex.m at lines 156-157 to store the confusion
   matrix.

3. Change the values of variables from lines 14 to 29 in ex.m.
   The changes must fulfill the following constraints:
        - percent <= 10
        - num_labels >= 100
        - num_unlabeled = 100*num_labels/percent;
        - input_layer_size  = 784;
        - hidden_layer_size = 100;
        - classes = 10;
        - 0 <= lambda <= 1
        - 0 < learning_rate < 0.1
        - 0 < momentum < 1
        - epochs >= 1000

4. Optional: Modify the architecture of the neural network 
   (e.g. adding more hidden layers)

5. Prepare the report with the plots obtained at point 1 
   and 2 for each experiment, commenting the obtained
   results and motivating the need of each experiment.

----------------------- DEADLINE -----------------------------

Submit the report and the code on the moodle by 5/1/2017.