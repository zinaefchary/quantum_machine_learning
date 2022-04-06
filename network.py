# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:45:47 2022

@author: Zina Efchary
"""
"""
network.py
~~~~~~~~~~
This code is based on Micheal Nielsen's classification algorithm, accessible at:
https://github.com/mnielsen/neural-networks-and-deep-learning
A module to implement the stochastic gradient descent learning algorithm for a
feedforward neural network.  Gradients are calculated using backpropagation.
Note that I have focused on making the code simple, easily readable, and easily
modifiable.  It is not optimized, and omits many desirable features.
"""

#### Libraries
# Standard library
import random


# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# Reading and converting data
import mnist_loader

# Hyperparamteres

EPOCHS = 5
BATCH_NUMEBR = 10
LEARNING_RATE = 3.0

# Defining global arrays to track the parameters through iterations

ACTIVATIONS = []
WEIGHTS = []
BIASES = []


def main():
    """
     Main function, executes the programm.

    Returns
    -------
    None.

    """


    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 4, 2])
    net.SGD(training_data, EPOCHS, BATCH_NUMEBR, LEARNING_RATE, test_data=test_data)
    activation_a, activation_b, activation_c, activation_d = find_activation(training_data)
    wavefunction, normalisation = get_wavefunction(activation_a, activation_b,
                                                   activation_c, activation_d)
    plot_graphs(activation_a, activation_b, activation_c, activation_d,
                wavefunction, normalisation)


def find_activation(training_data):
    """
    This function creats four arrays of the activation of the four neurons
    in the hidden layer after each iteration for the first image in the training
    set. This activation is equivalent to the psi_i and is the indivdual
    wavefunction of each neuron.

    Parameters
    ----------
    training_data : TYPE
        DESCRIPTION.

    Returns
    -------
    activation_a : TYPE
        DESCRIPTION.
    activation_b : TYPE
        DESCRIPTION.
    activation_c : TYPE
        DESCRIPTION.
    activation_d : TYPE
        DESCRIPTION.

    """
    # Saving a list of the activation for each neuron in the second
    # layer for the first image in the data set after each epoch

    counter = 0
    n = 0
    activation_a = []
    activation_b = []
    activation_c = []
    activation_d = []

    while counter < EPOCHS:
        n = len(training_data)*counter
        activation_a.append(ACTIVATIONS[n][0][0][0])
        activation_b.append(ACTIVATIONS[n][0][1][0])
        activation_c.append(ACTIVATIONS[n][0][2][0])
        activation_d.append(ACTIVATIONS[n][0][3][0])
        counter = counter + 1

    #print(WEIGHTS[0][0][0])
    #print(np.sign(WEIGHTS[0][0][0]))


    return activation_a, activation_b, activation_c, activation_d

def get_wavefunction(activation_a, activation_b, activation_c, activation_d):
    """


    Parameters
    ----------
    training_data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # assuming all four states are equally likely

    normalisation = 0.5

    wavefunction = normalisation*(np.add(np.add(activation_a, activation_b),
                                  np.add(activation_c, activation_d)))

    return wavefunction, normalisation

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # Introducing sign-entanglement between neuron a and c approach 1

        sign_weight_a = np.sign(self.weights[0][0][0])
        self.weights[0][2][0] = self.weights[0][2][0]*sign_weight_a

        # Introducing sign-entanglement between neuron a and c approach 2
        #unupdated_weights = []
        #unupdated_weights.append(self.weights)

        #n=0

       # while n <= len(unupdated_weights[0][0][0]):
       #     if self.weights[0][0][0]*sign_weight_a > 0:
       #         self.weights[0][2][0] = -self.weights[0][2][0]
       #     n = n + 1



        ACTIVATIONS.append(zs)
        WEIGHTS.append(self.weights)
        BIASES.append(self.biases)
        #print(self.weights)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def plot_graphs(activation_a, activation_b, activation_c, activation_d,
                wavefunction, normalisation):
    """ """

    epoch = np.linspace(1, EPOCHS, num=EPOCHS)

    # activation_a vs epoch

    plt.subplot(221)
    plt.plot(epoch , activation_a, color='blue')
    plt.scatter(epoch , activation_a, color='red')
    plt.title('neuron a')
    plt.ylabel("activation")
    plt.xticks(color='w')

    # activation_b vs epoch

    plt.subplot(222)
    plt.plot(epoch , activation_b, color='orange')
    plt.scatter(epoch , activation_b, color='red')
    plt.title('neuron b')
    plt.xticks(color='w')

    # activation_c vs epoch

    plt.subplot(223)
    plt.plot(epoch , activation_c, color='red')
    plt.scatter(epoch , activation_c, color='red')
    plt.title('neuron c')
    plt.xlabel("number of iterations")
    plt.xticks(range(1, EPOCHS + 1))
    plt.ylabel("activation")

    # activation_d vs epoch

    plt.subplot(224)
    plt.plot(epoch , activation_d, color='green')
    plt.scatter(epoch , activation_d, color='red')
    plt.title('neuron d')
    plt.xlabel("number of iterations")
    plt.xticks(range(1, EPOCHS + 1))

    # wavefunction vs epoch

    plt.figure()
    plt.plot(epoch, wavefunction)
    plt.scatter(epoch , wavefunction, color='red')
    plt.xlabel("number of iterations")
    plt.ylabel("total wavefunction")
    plt.xticks(range(1, EPOCHS + 1))
    plt.show()


# Main Code

if __name__== "__main__":
    main()


