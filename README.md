# CS6910_Assignment-1
Author: Sumit Sharma, EE20D042,

The task is to implement a Feed-forward Neural Network, and write a backpropogation code for training the network. This network will be trained and tested using the Fashion-MNIST dataset. 
Specifically, given an input image i.e., 28 x 28 = 784 pixels from the Fashion-MNIST dataset, the network will be trained to classify the image into 1 of 10 classes. 

In this Github repository, a Feed-forward Neural Network is implemented in a single google colab file (attached). 
Notations: 
inn----Input neuron,
otn----Output neuron,
hls----Hidden layers,
hn-----Hidden neuron,
xtr----Xtrain,
ytr----Ytrain,
xte----Xtest,
yte----Ytest, 
W---Weight Dictionary
b---Bias Dictionary
u---Dictionary for defining input to neuron
v---Dictionary for defining output from neuron
la---"la"th index for training data

The whole code is divided into different functions. The main functions are as follows: 
1. activ function----For calling different activation functions i.e. Sigmoid, relu and tanh,

2. deractiv function----For calling derivative activation function in order to get different derivatives for different activation functions,

3. OUT(ul, otn)----Output function i.e., the softmax function for the last layer of the network, that makes the outputs from different neurons as predictions/probabilities

4. forwardprop(xtr, hls, hn, inn, otn, W, b, u, v, la, activation)--- For defining the Forward Propogation step, which defines the forward pass, which gives the prediction,  

5. init(W, b, inn, hls, hn, otn, initialization)---For initialization for the weights and biases in which the argument initialization decides regarding the initialization method i.e., "random" or "xavier". 

6. hot(m, n): Defining the hot function in order to provide "hot vectors" i.e. each class gets uniquely identified with different encodings

7. Backprop(ytr, hls, hn, inn, otn, W, b, u, v, gradW, gradb, la, y_predicted, activation, los): For defining the backpropogation step, in order to find the gradients of loss with respect to weights and biases

8. LOSS(y_predicted, ytr, los, la, otn)---For defining the loss i.e., Squared error and Cross entropy loss, this function takes "5" arguments. 

9. acc(xte, yte, xtr, ytr, W, b, hls, otn, hn, activation)---This is the accuracy function, which takes "10" arguments and responsible for returning the validation accuracy with confusion matrix

The different optimization methods definitions start in terms of different functions
10. momentumbasedgd(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, set_train, activation, los)--Momentumbased gradient descent method

11. RMS(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, set_train, activation, los)--Root-mean-sqaure propogation method

12. Nesterovagd(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, set_train, activation, los)--Nesterov gradient descent method

13. stochasticgd(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, set_train, activation, los)--Stocastic gradient descent method

14. Nesterovadapt(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, set_train, activation, los)--Nesterov adaptive movement estimation method

15. adaptivem(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, set_train, activation, los)--Adaptive movement estimation method

16. trning(xtr, ytr, hls, hn, inn, otn, W, b, u, v, learning_rate, epochs, size_batch, Optimization, set_train, activation, los, initialization)---Finally the training function with "18" arguments as input to call different functions. 

For sweep through wandb, these are the range for different parametres----
sweep_config = {
    'name'  : "Sumit Sharma", 
    'method': 'random', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },
 'parameters': {

        'hls': {
            'values': [3,4, 5]
        },
        'epochs': {
            'values': [10, 15]
        },
        'hn': {
            'values': [32, 64,128]
        },
        'learning_rate': {
            'values': [1e-2,1e-3,5e-3]
        },
        'initialization': {
            'values': ["random","xavier"]
        },
        'size_batch': {
            'values': [32,64]
        },
        'Optimization': {
            'values': ["momentumbasedgd","stochasticgd","Nesterovadapt","RMS","adaptivem","nesterovagd"]
        },
        'activation': {
            'values': ["sigmoid","relu","tanh"]
        }
    }
}
For training, 90% dataset selected i.e. 54000 samples and 10% dataset as test dataset.
No of output neurons selected as "10". 
No of input neurons selected as "784".
"los" variable set as "cross_entropy"
Finally there is an execute(), to call the "trning()" and to initiate wandb


