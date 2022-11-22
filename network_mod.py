# -*- coding: utf-8 -*-
"""
-----------------QNN-----------------------------

-------------------------------------------------
This code is based on Micheal Nielsen's classification algorithm, accessible at:
https://github.com/mnielsen/neural-networks-and-deep-learning

The classical code involves a module to implement the stochastic gradient
descent learning algorithm for a feedforward neural network. Gradients are
calculatedusing backpropagation.Modifications have been made to include
quantum-behaviour in the network. The resulting behaviour is plotted.
Last Updated: 06/05/22
@Author: Zina Efchary, Ksenija Kovalenka
"""

#### Libraries
# Standard libraries
import random
import numpy as np
import matplotlib.pyplot as plt

# Reading and converting data
import mnist_loader

# for activation functions
from definitions import sigmoid, sigmoid_prime, superposition
from scipy.misc import derivative

#CONSTANTS

# Hydrogen radial wavefunction parameters
NL = np.array([[1, 0],[2, 0],[2, 1],[3, 0],[3, 1],[3, 2]])

# Learning (stocastic gradient decent) parameters
EPOCHS = 10
BATCH_NUMEBR = 10
LEARNING_RATE = 0.5

# activation for classical neurons
CLASSICAL_ACTIVATION = sigmoid
CLASSICAL_ACTIVATION_PRIME = sigmoid_prime

# entanglement switch 
CNOT = False


def main():
    """
     Main function, executes the programm.

    Returns
    -------
    None.

    """
    redefine_globals()
    # trian
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 6, 3, 2])
    net.SGD(training_data, EPOCHS, BATCH_NUMEBR, LEARNING_RATE, test_data=test_data)
    
    # 3D plot
    activations, results, out_layer, inputs = find_activations3D(training_data)
    normalised_mag_field3D = get_mag_field3D(activations)
    # raiting calculation
    randomness = plot_vectors3D(normalised_mag_field3D, results, out_layer)
    print('randomness parameter: {}'.format(randomness))
    print('Typycal parameters: 0.6 - 0.8 for uniform random distribution')
    print('                    0 for no randomness')

    # activation plot in quantum layer
    plot_progress(inputs, activations)


def redefine_globals():
    
    # intiallising global arrays for the new run
    
    # needed for plotting
    global ACTIVATIONS, DESIRED_OUTCOME, INPUTS
    ACTIVATIONS = []
    DESIRED_OUTCOME = []
    INPUTS = []
    # might be needed for plotting
    # global WEIGHTS, BIASES
    # WEIGHTS = []
    # BIASES = []
    
    # initialising new weights for the quantum function
    global WEIGHT, FUNCTION, FUNCTION_PRIME, QUANTUM_ACTIVATION, QUANTUM_ACTIVATION_PRIME
    
    # # only positive coeffs run 1 randomness: meh learning: meh
    # WEIGHT = np.array([0.60081542, 0.95352262, 0.99802231, 0.75889375, 1.50197233, 0.8167927])
    # all coeffs run 1 randomness: cool learning: meh
    # WEIGHT = np.array([-0.43716806, 0.75794288, -0.34619758, 0.62708196, -0.03037096, 0.26817109])
    # only positive coeffs run 2 randomness: cool learning: cool (surprisingly)
    WEIGHT = np.array([1.46161031, 1.33253892, 1.4240542, 1.13220447, 1.16093561, 1.37562307])
    # # all coeffs run 2 randomness: meh learning: meh
    # WEIGHT = np.array([0.65750386, 0.04267012, -0.10281869, -0.71354464, 0.10293271, -0.25391892])
    
    def FUNCTION(a, nl=NL, weight_arr=WEIGHT):
        
        
        activation = np.where(a >= 0, superposition(a, nl_array=nl, weight_array=weight_arr), 
                        superposition(-a, nl_array=nl, weight_array=weight_arr))
    
        global CNOT
        #cnot
        if activation[2] > 0: CNOT = True
        #flipping the next qubit
        if CNOT:
            for i in range(len(activation)-3):
                activation[i+3] = -activation[i+3]        
        #initiallise again
        CNOT = False
        
        return activation
    
    def FUNCTION_PRIME(a):
        return derivative(FUNCTION, a, 1e-6)
    
    QUANTUM_ACTIVATION = FUNCTION
    QUANTUM_ACTIVATION_PRIME = FUNCTION_PRIME

        
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
        bw = [list(x) for x in zip(self.biases, self.weights)]
        
        # gaussian activation for quantum layer
        a = QUANTUM_ACTIVATION(np.dot(bw[0][1], a)+bw[0][0])
        
        #normal activation for the rest of the network
        for b, w in bw[1:]:
            a = CLASSICAL_ACTIVATION(np.dot(w, a)+b)
            
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
        
        bw = [list(x) for x in zip(self.biases, self.weights)]
        # quantum layer
        z = np.dot(bw[0][1], activation)+bw[0][0]
        zs.append(z)
        activation = QUANTUM_ACTIVATION(z)
        activations.append(activation)
        
        # the rest of the network
        for b, w in bw[1:]:
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = CLASSICAL_ACTIVATION(z)
            activations.append(activation)
        
        # backward pass of the last layer
        delta = self.cost_derivative(activations[-1], y) * \
            CLASSICAL_ACTIVATION_PRIME(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #the rest of the normal layers
        for l in range(2, self.num_layers):
            
            activation_prime = CLASSICAL_ACTIVATION_PRIME
            # for last quantum layer (last from the point of view of backprop)
            if l == self.num_layers:
                activation_prime = QUANTUM_ACTIVATION_PRIME
            z = zs[-l]
            sp = activation_prime(z)
            
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            

        # adding the resulting activation and paramters to the global array
        DESIRED_OUTCOME.append(y)
        ACTIVATIONS.append(activations)
        INPUTS.append(zs)
        # WEIGHTS.append(self.weights)
        # BIASES.append(self.biases)
        

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
    
    
#-----------------------------3D_code---------------------------------------------

def find_activations3D(training_data):
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
    
    activations = []
    results = []
    out_layer = []
    inputs = []

    while counter < EPOCHS:
        n = len(training_data)*counter
        # taking activations from the 2nd (quantum) layer
        inputs.append(INPUTS[n][0])
        activations.append(ACTIVATIONS[n][1])
        # storing network output and desired output    
        # CAREFULL, WON'T WORK WITH 9 INPUTS
        results.append(DESIRED_OUTCOME[n][1])
        out_layer.append(ACTIVATIONS[n][-1])

        counter = counter + 1
    activations = np.array(activations)
    activations = activations.reshape(EPOCHS, len(ACTIVATIONS[n][1]))
    
    inputs = np.array(inputs)
    inputs = inputs.reshape(EPOCHS, len(INPUTS[n][0]))

    return activations, results, out_layer, inputs
    


def get_mag_field3D(activations):
    """
    This function converts the activations of the neurons in the ssecond layer
    into a vector describing the orientation of the spin of an electron in the
    magnetic field. Subsequantly, the length of the vector is normalised.
    """
    #set up magnetic field equal to activations
    mag_field = activations
    # initialise final magnetic field array
    normalised_mag_field3D = np.empty(0)
    # separate the array into individual qubits
    
    for i in range(len(mag_field[:,0])):
        for j in np.arange(0, len(mag_field[0]), 3):
            
            normalised_mag_field3D_vector = mag_field[i, j:j+3]/np.linalg.norm(mag_field[i, j:j+3])
            normalised_mag_field3D = np.append(normalised_mag_field3D, normalised_mag_field3D_vector)
        
    normalised_mag_field3D = normalised_mag_field3D.reshape(np.shape(mag_field))    

    return normalised_mag_field3D


    
def plot_vectors3D(normalised_mag_field3D, results, out_layer):
    
    # draw sphere
    U, V = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    X = np.cos(U)*np.sin(V)
    Y = np.sin(U)*np.sin(V)
    Z = np.cos(V)


    # import 3D axes
    from mpl_toolkits.mplot3d import Axes3D
    plt.rcParams.update({'font.size': 14})
    plt.style.use('default')

    # draw a vector
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0,0), (0,0), *args, **kwargs)
            self._verts3d = xs, ys, zs
    
        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
            self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
    
            return np.min(zs)
               

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=270)
    ax.plot_surface(X, Y, Z, alpha=0.2)
    ax.set_title('Quantum Layer Plot')
    

    color = iter(plt.cm.PuRd(np.linspace(0, 1, len(normalised_mag_field3D[:,0]))))
    
    for i in range(len(normalised_mag_field3D[:,0])):
        
        # if results[i] == np.argmax(out_layer[i]):
        #     c = 'g'
        # else:
        #     c = 'r'
        c = next(color)
        
        for j in np.arange(0, len(normalised_mag_field3D[0]), 3):
        # for j in np.arange(0, 1, 3):
            if normalised_mag_field3D[i,2] > 0:
                c = 'blue'
                if j != 0:
                    c = 'navy'
            arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color=c, alpha=0.9)
            a = Arrow3D([0, normalised_mag_field3D[i,j]], 
                        [0, normalised_mag_field3D[i,j+1]], 
                        [0, normalised_mag_field3D[i,j+2]], **arrow_prop_dict)
            ax.add_artist(a)
            
    # calculating randomness parameter 
    total_rating = calculate_randomness(normalised_mag_field3D)
    
    # fix labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.savefig('latest_run.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return total_rating

def calculate_randomness(vector_array):
    
    vector_array = np.reshape(vector_array, (EPOCHS*2, 3))
    # calculating randomness parameter
    mean = []
    for i in range(len(vector_array[0])):
        mean.append(np.mean(vector_array[:,i]))

    new_mean=[]
    for i in range(len(vector_array[0])):
        new_mean.append(np.mean(vector_array[:EPOCHS,i]))

    mean = np.array(mean)
    new_mean = np.array(new_mean)
    mean_difference = np.linalg.norm(mean - new_mean)
    
    variance = []
    for i in range(len(vector_array[0])):
        variance.append(np.var(vector_array[:,i]))

    randomness_parameter_std = np.sqrt(sum(variance))
    
    total = randomness_parameter_std - mean_difference
    
    if total < 0:
        total = 0
    
    return total
    

def plot_progress(inputs, activations):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    X = np.arange(-20, 20, 0.1)
    ax.plot(X, QUANTUM_ACTIVATION(X), c='k')
    
    color = iter(plt.cm.PuRd(np.linspace(0, 1, len(inputs[:,0]))))
    
    for i in range(len(inputs[:,0])):
        c = next(color)
        if activations[i, 2] > 0:
            c = 'blue'
        ax.scatter(inputs[i], activations[i], c=c)
    

# Main Code

if __name__== "__main__":
    main()
