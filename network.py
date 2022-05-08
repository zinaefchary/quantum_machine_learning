# -*- coding: utf-8 -*-
"""
-----------------TITLE---------------------------
PHYS30880 -  BSc Dissertation - network.py
-------------------------------------------------
This code is based on Micheal Nielsen's classification algorithm, accessible at:
https://github.com/mnielsen/neural-networks-and-deep-learning

The classical code involves a module to implement the stochastic gradient
descent learning algorithm for a feedforward neural network. Gradients are
calculatedusing backpropagation.Modifications have been made to include
quantum-behaviour in the network. The resulting behaviour is plotted.
Last Updated: 06/05/22
@Author: Zina Efchary
"""

#### Libraries
# Standard library
import random


# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from math import atan, pi



# Reading and converting data
import mnist_loader

# Hyperparamteres

EPOCHS = 10
BATCH_NUMEBR = 10
LEARNING_RATE = 3.0

# Defining global arrays used to track the change of parameters through epochs

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
    net = Network([784, 2, 2])
    net.SGD(training_data, EPOCHS, BATCH_NUMEBR, LEARNING_RATE, test_data=test_data)
    activation_a, activation_b = find_activation(training_data)
    normalised_mag_field = get_mag_field(activation_a, activation_b)
    plot_vectors(normalised_mag_field)

def find_activation(training_data):
    """
    This function creats four arrays of the activation of the four neurons
    in the hidden layer after each iteration for the first image in the training
    set. This activation is equivalent to the psi_i and is the indivdual
    wavefunction of each neuron.

    """
    # Saving a list of the activation for each neuron in the second
    # layer for the first image in the data set after each epoch

    counter = 0
    n = 0
    activation_a = []
    activation_b = []

    while counter < EPOCHS:
        n = len(training_data)*counter

        activation_a.append(ACTIVATIONS[n][1][0])
        activation_b.append(ACTIVATIONS[n][1][1])

        counter = counter + 1


    return activation_a, activation_b


def get_mag_field(activation_a, activation_b):
    """
    This function converts the activations of the neurons in the ssecond layer
    into a vector describing the orientation of the spin of an electron in the
    magnetic field. Subsequantly, the length of the vector is normalised.
    """

    # defining the magnetic field components and creating the correspending
    # array

    mag_x = activation_a
    mag_y = activation_b
    mag_field = np.hstack((mag_x, mag_y))
    #print(mag_field)

    # finding the normalisation of each vector in the array and normalising it

    normalisation_array = []
    normalised_mag_field = []
    n = 0

    while n < len(mag_field):

        normalisation_array.append(np.linalg.norm(mag_field[n]))
        normalised_mag_field.append(mag_field[:][n] / normalisation_array[n])
        n = n + 1

    # converting the resulting tuple back into an array

    normalised_mag_field = np.asarray(normalised_mag_field)

    return normalised_mag_field


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
            a = gaussian(np.dot(w, a)+b)
        return a
    def feedforward2(self, a):

        zs = []
        activations = [a]

        activation = a
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = gaussian(z)
            activations.append(activation)

        return (zs, activations)

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
    def update_mini_batch2(self, mini_batch, eta):
        batch_size = len(mini_batch)

        # transform to (input x batch_size) matrix
        x = np.asarray([_x.ravel() for _x, _y in mini_batch]).transpose()
        # transform to (output x batch_size) matrix
        y = np.asarray([_y.ravel() for _x, _y in mini_batch]).transpose()

        nabla_b, nabla_w = self.backprop2(x, y)
        self.weights = [w - (eta / batch_size) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / batch_size) * nb for b, nb in zip(self.biases, nabla_b)]

        return


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
            activation = gaussian(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            gaussian_prime(zs[-1])
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
            sp = gaussian_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        # adding the resulting activation and paramters to the global array

        ACTIVATIONS.append(activations)
        WEIGHTS.append(self.weights)
        BIASES.append(self.biases)

        return (nabla_b, nabla_w)

    def backprop2(self, x, y):

        nabla_b = [0 for i in self.biases]
        nabla_w = [0 for i in self.weights]

        # feedforward
        zs, activations = self.feedforward2(x)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * gaussian_prime(zs[-1])
        nabla_b[-1] = delta.sum(1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
           z = zs[-l]
           sp = gaussian_prime(z)
           delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
           nabla_b[-l] = delta.sum(1).reshape([len(delta), 1]) # reshape to (n x 1) matrix
           nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

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

#### activation functions functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def gaussian(z):
    """A Gaussian-like function."""
    return (np.exp((-(z + 0.1)**2)/10)+ np.exp((-(z - 0.)**2)/10))/100

def gaussian_prime(z):
    """Derivative of gaussian function."""
    return 0.01*(-2*(z+0.1)/10)*np.exp(-(z+0.1)**2/10) - 0.01*(2*(z-0.1)/10)*np.exp(-(z-0.1)**2/10)

def relu(z):
    """The relu function."""
    if z.any() < 0:
        z = 0
    else:
        z = z
    return z

def relu_prime(z):
    """Derivative of relu function."""
    if z.any() <= 0:
        z = 1
    else:
        z = 0
    return z

def plot_vectors(normalised_mag_field):
    """
    This function converts the spin orientation of the virtual Q-bit into polar
    coordinates and subsequently, plots it.
    """

    # defining the coordinates of magnetic fields in polar coordinates

    r = []
    theta = []
    labels = []
    n = 0

    # finding the angles by converting it from cartesian coordinates to polar

    while n < len(normalised_mag_field):
        r.append(np.linalg.norm(normalised_mag_field[n]))
        if normalised_mag_field[n][0] > 0 and normalised_mag_field[n][1] > 0:
            theta.append(atan(normalised_mag_field[n][1]/normalised_mag_field[n][0]))
        if normalised_mag_field[n][0] < 0:
            theta.append((np.pi + atan(normalised_mag_field[n][1]/normalised_mag_field[n][0])))
        if normalised_mag_field[n][0] > 0 and normalised_mag_field[n][1] < 0:
            theta.append(( 2*np.pi + atan(normalised_mag_field[n][1]/normalised_mag_field[n][0])))
        n = n + 1

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, projection = 'polar')

    # plotting the vectors

    num_plots = len(normalised_mag_field)
    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.OrRd(np.linspace(0, 1, num_plots))))

    for i in range(len(r)):
        ax.plot([0, theta[i]], [0, r[i]])
        labels.append(r'epoch %i' % (i))


    # colouring the area of classification

    specifity = np.pi / 8
    angle_1 = np.linspace(3*np.pi / 8 - specifity, 3*np.pi / 8 + specifity, 100)
    angle_0 = np.linspace(np.pi / 8 - specifity, np.pi / 8 + specifity, 100)
    ax.fill_between(angle_1, 0, 1, color='r', alpha=0.1, label='classified as 1')
    ax.fill_between(angle_0, 0, 1, color='b', alpha=0.1, label='classified as 0')


    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

    # creating a legend

    legend1 = ax.legend(loc='upper left', bbox_to_anchor=(0.5, -0.05))
    ax.legend(labels, loc='lower right', bbox_to_anchor=(0.5, -0.25), ncol= 2)
    ax.add_artist(legend1)

    plt.show()

# Main Code

if __name__== "__main__":
    main()


