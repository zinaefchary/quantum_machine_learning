# Quantum Machine Learning

Quantum phenomena such as entanglement, tunnelling or superposition are
increasingly used in quantum information to perform computation. Moreover,
machine learning methods are driving a paradigm shift in the field of artificial
intelligence. This project aims to explore the possibility of implementing quantum
phenomena into machine learning algorithms. A theoretical approach to achieve
this is proposed by modifying the architecture of a classical image recognition
neural network such that it imitates the behaviour of a single Q-bit in a magnetic
field. The resulting system is a quantum-inspired neural network does a reasonable job in classifying handwritten digits of 0 and 1.

# General Architecture of the proposed Quantum-Inspired-Neural Network

![architecture (0) (1) (1)](https://user-images.githubusercontent.com/99489418/167292825-b1511663-e7e6-4240-87b5-fbd3e4ba88ca.png)

## a) The structure of the NN 
Note that the input is either a 28x28 pixel handwritten image
of 1 or 0. The neurons in the hidden layer 1 correspond to the magnetisation direction of
the virtual Q-bit. The neurons in the hidden layer 2 represent the measurement operators.
## b) Quantum equivalent of the network 
Firstly, the virtual Q-bit is being prepared and
subsequently measured. The result of the measurement corresponds to either 1 or 0. Note
that the entire system runs on a classical computer but introduces physical/quantum-
mechanical phenomena into the ML algorithm.

### Referenec: The code is based on the work done by Michael Nielsen, accessible at: https://github.com/mnielsen/neural-networks-and-deep-learning
Update: Further improvements and modifications have been added in collaboration with Ksenija Kovalenka.
