import numpy as np
import math


class NeuralNetwork:
    """
    A multi-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices.

    The network uses a nonlinearity after each fully connected layer except for the
    last. You will implement two different non-linearities and try them out: Relu
    and sigmoid.

    The outputs of the second fully-connected layer are the scores for each class.
    """


    def __init__(self, input_size, hidden_sizes, output_size, num_layers, nonlinearity='relu'):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H_1)
        b1: First layer biases; has shape (H_1,)
        .
        .
        Wk: k-th layer weights; has shape (H_{k-1}, C)
        bk: k-th layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: List [H1,..., Hk] with the number of neurons Hi in the hidden layer i.
        - output_size: The number of classes C.
        - num_layers: Number of fully connected layers in the neural network.
        - nonlinearity: Either relu or sigmoid
        """
        self.num_layers = num_layers

        assert(len(hidden_sizes)==(num_layers-1))
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params['W' + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            print('init')
            self.params['b' + str(i)] = np.zeros(sizes[i])

        if nonlinearity == 'sigmoid':
            self.nonlinear = sigmoid
            self.nonlinear_grad = sigmoid_grad
        elif nonlinearity == 'relu':
            self.nonlinear = relu
            self.nonlinear_grad = relu_grad


    def forward(self, X):
        """
        Compute the scores for each class for all of the data samples.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.

        Returns:
        - scores: Matrix of shape (N, C) where scores[i, c] is the score for class
            c on input X[i] outputted from the last layer of your network.
        - layer_output: Dictionary containing output of each layer BEFORE
            nonlinear activation. You will need these outputs for the backprop
            algorithm. You should set layer_output[i] to be the output of layer i.

        """

        scores = None
        layer_output = {}
        #############################################################################
        # TODO: Write the forward pass, computing the class scores for the input.   #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C). Store the output of each layer BEFORE nonlinear activation  #
        # in the layer_output dictionary                                            #
        #############################################################################

        # Implement two seperate cases: 2-layer NN and 3-layer NN
        if self.num_layers == 2: 
            z = X.dot(self.params['W1']) + self.params['b1'] # Z = X.W1 + b1
            layer_output[1] = z
            z = self.nonlinear(z)
            z2 = z.dot(self.params['W2']) + self.params['b2'] # Z2 = Z.W2 + b2
            layer_output[2] = z2
            scores = z2                                       # return scores
        elif self.num_layers == 3: 
            z = X.dot(self.params['W1']) + self.params['b1']  # Z = X.W1 + b1
            layer_output[1] = z
            z_t = self.nonlinear(z)
            z2 = z_t.dot(self.params['W2']) + self.params['b2']  # Z2 = Z.W2 + b2
            layer_output[2] = z2
            z2_t = self.nonlinear(z2)
            z3 = z2_t.dot(self.params['W3']) + self.params['b3'] # Z3 = Z.W3 + b3
            layer_output[3] = z3
            #z3_t = self.nonlinear(z3) --> do we need this??
            scores = z3
        return scores, layer_output


    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """

        # Compute the forward pass
        # Store the result in the scores variable, which should be an array of shape (N, C).
        scores, layer_output = self.forward(X)

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss using the scores      #
        # output from the forward function. The loss include both the data loss and #
        # L2 regularization for weights W1,...,Wk. Store the result in the variable #
        # loss, which should be a scalar. Use the Softmax classifier loss.          #
        #############################################################################

        ### Reference: http://cs231n.github.io/neural-networks-case-study/ ###### --> Very useful for understanding loss functions

        #get nexessary shapes
        N = scores.shape[0]        
        C = scores.shape[1]

        data_loss = 0
        mu = 0.5

        # Apply the softmax loss function
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims=True)
        correct_logprobs = -np.log(probs[range(N), y])
        data_loss = np.sum(correct_logprobs) / N

        # two seperate cases for reg_loss 
        if self.num_layers == 2:
            reg_loss = mu * reg * np.sum(self.params['W1'] * self.params['W1']) + mu * reg * np.sum(self.params['W2'] * self.params['W2']) 

        if self.num_layers == 3: 
            reg_loss = mu * reg * np.sum(self.params['W1'] * self.params['W1']) + mu * reg * np.sum(self.params['W2'] * self.params['W2']) +  mu * reg * np.sum(self.params['W3'] * self.params['W3'])

        #Cross entropy loss
        loss = data_loss + reg_loss
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients
        grads = {}
        #############################################################################
        # TODO: Compute the backward pass, computing the derivatives of the weights #
        # and biases. Store the results in the grads dictionary. For example,       #
        # grads['W1'] should store the gradient on W1, and be a matrix of same size #
        #############################################################################
        grad_scores = probs
        N,D = X.shape
        grad_scores[np.arange(N),y] -= 1
        grad_scores = grad_scores / N

        #print(layer_output[1].shape) #(5,10)
        #print(layer_output[2].shape) #(5,3)
        #print(grad_scores.shape) #(5,3)
        #print(self.params['W2'].shape) #(10,3)
        #print(X.shape) #(5,4)


        # two seperate cases for backpropagation #
        if self.num_layers == 2:
            grads['W2'] = (np.dot(self.nonlinear(layer_output[1].T),grad_scores))       # dW2 = g(z1).dscores
            grads['b2'] = np.sum(grad_scores, axis = 0, keepdims = True)                # sum dscores to get bias

            hidden_layer1 = (grad_scores.dot(self.params['W2'].T))                      # hidden layer = dscores.W2 * g(z1)
            hidden_layer1 *= (self.nonlinear_grad(layer_output[1]))

            grads['W1'] = np.dot(X.T,(hidden_layer1)) 
            grads['b1'] = np.sum(grad_scores.dot(self.params['W2'].T)*(self.nonlinear_grad(layer_output[1])),axis = 0, keepdims = True)

            grads['W2'] = (reg * self.params['W2']) + grads['W2']                       #regularization
            grads['W1'] = (reg * self.params['W1']) + grads['W1']
            

        if self.num_layers == 3: 
            grads['W3'] = np.dot(self.nonlinear(layer_output[2].T),grad_scores)        # same formula as two layers, but just add W3 and one more hidden layer
            grads['b3'] = np.sum(grad_scores, axis=0, keepdims=True)

            hidden_layer1 = (grad_scores.dot(self.params['W3'].T))
            hidden_layer1 *= (self.nonlinear_grad(layer_output[2]))

            grads['W2'] = np.dot(self.nonlinear(layer_output[1]).T,hidden_layer1)
            grads['b2'] = np.sum(hidden_layer1, axis=0, keepdims=True)

            hidden_layer2 = hidden_layer1.dot(self.params['W2'].T)
            hidden_layer2 *= (self.nonlinear_grad(layer_output[1]))

            grads['W1'] = np.dot(X.T,(hidden_layer2))
            grads['b1'] = np.sum(hidden_layer2, axis = 0, keepdims = True)

            grads['W3'] = (reg * self.params['W3']) + grads['W3']
            grads['W2'] = (reg * self.params['W2']) + grads['W2']
            grads['W1'] = (reg * self.params['W1']) + grads['W1']
        

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)
        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            #########################################################################
            # TODO: Create a random minibatch of training data and labels, storing  #
            # them in X_batch and y_batch respectively.                             #
            #########################################################################
            random_idx = np.random.choice(np.arange(X.shape[0]), batch_size)
            X_batch = X[random_idx]
            y_batch = y[random_idx]
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            #########################################################################
            # TODO: Use the gradients in the grads dictionary to update the         #
            # parameters of the network (stored in the dictionary self.params)      #
            # using stochastic gradient descent. You'll need to use the gradients   #
            # stored in the grads dictionary defined above.                         #
            #########################################################################
            if self.num_layers == 2: 
                self.params['W2'] = -1 * learning_rate * grads['W2'] + self.params['W2']
                self.params['b2'] = -1 * learning_rate * grads['b2'] + self.params['b2']
                self.params['W1'] = -1 * learning_rate * grads['W1'] + self.params['W1']
                self.params['b1'] = -1 * learning_rate * grads['b1'] + self.params['b1']
            if self.num_layers == 3: 
                self.params['W3'] = -1 * learning_rate * grads['W3'] + self.params['W3']
                self.params['b3'] = -1 * learning_rate * grads['b3'] + self.params['b3']
                self.params['W2'] = -1 * learning_rate * grads['W2'] + self.params['W2']
                self.params['b2'] = -1 * learning_rate * grads['b2'] + self.params['b2']
                self.params['W1'] = -1 * learning_rate * grads['W1'] + self.params['W1']
                self.params['b1'] = -1 * learning_rate * grads['b1'] + self.params['b1']
            #########################################################################
            #                             END OF YOUR CODE                          #
            #########################################################################
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        y_pred = None

        ###########################################################################
        # TODO: Implement classification prediction. You can use the forward      #
        # function you implemented                                                #
        ###########################################################################
        # compute the forward pass --> scores & find the predictions for class numbers
        y_pred = np.argmax(self.forward(X)[0],axis=1)
        return y_pred


def sigmoid(X):
    #############################################################################
    # TODO: Write the sigmoid function                                          #
    #############################################################################
    sig = (1 / (1 + np.exp(-X)))
    return sig


def sigmoid_grad(X):
    #############################################################################
    # TODO: Write the sigmoid gradient function                                 #
    #############################################################################
    return sigmoid(X) * (1 - sigmoid(X))


def relu(X):
    #############################################################################
    #  TODO: Write the relu function                                            #
    #############################################################################
    return np.maximum(0,X)


def relu_grad(X):
    #############################################################################
    # TODO: Write the relu gradient function                                    #
    #############################################################################
    X[X <= 0] = 0
    X[X > 0] = 1
    return X


