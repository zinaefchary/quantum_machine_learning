#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of functions and parameters for quantum neural network.

@author: ksenija kovalenka
"""
import numpy as np
import scipy.special as spe
from scipy.special import sph_harm
from scipy.misc import derivative

'''
# gaussian parameters: one tiiiiny gaussian 
WIDTH = 0.1
DEVIATION = 10

# gaussian parameters: one proper
WIDTH = 0.1
DEVIATION = 1.5

# gaussian parameters: two proper
WIDTH = 1.3 #u
DEVIATION = 0.7 #a

# gaussian parameters: three proper
WIDTH = 0.8 #u
DEVIATION = 1.2 #a
WIDTH_TWO = 1.2 #v
'''

# gaussian parameters: three proper
WIDTH = 0.1 #u
DEVIATION = 10 #a
WIDTH_TWO = 1.2 #v

# smooth parameters
STEEPNESS = 5
OFFSET = 10

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def gaussian(z):
    """A Gaussian-like function."""
    return (np.exp((-(z + WIDTH)**2)/DEVIATION)+ np.exp((-(z - WIDTH)**2)/DEVIATION))/DEVIATION**2

def gaussian_prime(z):
    """Derivative of gaussian function."""
    return ((-2*(z + WIDTH)/DEVIATION)*np.exp(-(z + WIDTH)**2/DEVIATION) - (2*(z - WIDTH)/DEVIATION)*np.exp(-(z - WIDTH)**2/DEVIATION))/DEVIATION**2

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    """Derivative of relu function."""
    return (z > 0).astype(int)

def gaussian_three(z):
    """A Gaussian-like function with three peaks"""
    return (np.exp((-(z - WIDTH - WIDTH_TWO)**2)/DEVIATION)+ np.exp((-(z + WIDTH + WIDTH_TWO)**2)/DEVIATION) + np.exp((-(-z + WIDTH - WIDTH_TWO)**2)/DEVIATION))/DEVIATION**2

def gaussian_three_prime(z):
    """Derivative Gaussian-like function with three peaks"""
    return ((-2*(z + WIDTH + WIDTH_TWO)/DEVIATION)*np.exp(-(z + WIDTH + WIDTH_TWO)**2/DEVIATION) - (2*(z - WIDTH - WIDTH_TWO)/DEVIATION)*np.exp(-(z - WIDTH - WIDTH_TWO)**2/DEVIATION) + (2*(-z - WIDTH + WIDTH_TWO)/DEVIATION)*np.exp(-(-z - WIDTH + WIDTH_TWO)**2/DEVIATION))/DEVIATION**2

def smooth_step(z):
    """
    smooth hat step
    """
    return (1/(1+np.exp(-STEEPNESS*(z+OFFSET))))-(1/(1+np.exp(-STEEPNESS*(z-OFFSET))))

def smooth_step_prime(z):
    return (STEEPNESS*np.exp(-STEEPNESS*(z+OFFSET)))/((np.exp(-STEEPNESS*(z+OFFSET))+1)**2) - (STEEPNESS*np.exp(-STEEPNESS*(z-OFFSET)))/((np.exp(-STEEPNESS*(z-OFFSET))+1)**2)

  


#------------------------------more_gaussians------------------------------------

# params
GAUSSIAN_NUMBER = 4
SEPARATION = 1.2

def gaussian_multiple(z, number = GAUSSIAN_NUMBER, 
                      separation = SEPARATION, 
                      height = 1):
    '''
    Many gaussians with controllanble number and separation.
    '''
    n = number
    s = separation
    # fix deviantion for now
    d = height

    y = sum([(np.exp((-(z + (2*i + 1)*s)**2)/d) + 
              np.exp((-(z - (2*i + 1)*s)**2)/d))/d**2 
             for i in np.arange(0, int(n/2))])
    return y  

def gaussian_multiple_prime(z, number = GAUSSIAN_NUMBER, 
                            separation = SEPARATION, 
                            height = 1):
    '''
    Many gaussians with controllanble number and separation derivative.
    '''
    n = number
    s = separation
    # fix deviantion for now
    d = height
    
    y = sum([((-2*(z + (2*i + 1)*s)/d)*np.exp((-(z + (2*i + 1)*s)**2)/d)+ 
              (-2*(z - (2*i + 1)*s)/d)*np.exp((-(z - (2*i + 1)*s)**2)/d))/d**2 
             for i in np.arange(0, int(n/2))])

    return y   

#------------------------------HYDROGEN------------------------------------

def psi_R(r,n=1,l=0):

    coeff = np.sqrt((2.0/n)**3 * spe.factorial(n-l-1) /(2.0*n*spe.factorial(n+l)))
    
    laguerre = spe.assoc_laguerre(2.0*r/n,n-l-1,2*l+1)
    
    return coeff * np.exp(-r/n) * (2.0*r/n)**l * laguerre

NL = np.array([[1, 0],[2, 0],[2, 1],[3, 0],[3, 1],[3, 2]])
WEIGHT = np.array([1, 1, 1, 1, 1, 1])

def superposition(r, nl_array, weight_array):
    
    return np.sum([weight * psi_R(r, nl[0], nl[1]) 
                for nl, weight in zip(nl_array, weight_array)], axis=0)

def superposition_prime(r):
    return derivative(superposition, r, 1e-6)







