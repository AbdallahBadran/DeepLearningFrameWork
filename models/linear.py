from numpy.random import randn
from numpy import dot, sum, zeros_like
from .data_and_grad import DataAndGrad


class Linear:  # It is the linear part of every layer #was class Linear(Function):
    def __init__(self, prevNodes, currNodes):  # flag i
        # Create weights and their data and gradient attributes
        self.W = DataAndGrad((prevNodes, currNodes))
        # Initializing the data of the weights
        self.W.data = 0.01 * randn(self.W.data.shape[0], self.W.data.shape[1])
        # Create weights and their data and gradient attributes

        # Create weights and their data and gradient attributes
        self.b = DataAndGrad((1, currNodes))
        # Initializing the data of the bias
        self.b.data = zeros_like(self.b.data)

    def forward(self, X):  # Here self is a layer   f.forward(X) f--> Z1
        # Implementing Z = X*W +b
        self.Z = dot(X, self.W.data) + self.b.data
        # Z will be of shape (number of instances in batch, number of nodes of current layer)
        self.input = X  # Storing the input to each layer "X" as this is needed to compute the gradients of W in
        # the backward function.
        return self.Z

    def backward(self, dL_dZ):
        # Gradient of W :
        self.dZ_dW = self.input.T
        self.dL_dW = dot(self.dZ_dW, dL_dZ)  # dL/dW (for all samples) = (dL/dZ)*(dZ/dW)
        self.W.grad += self.dL_dW  # Accumulating on the W gradient  #delta
        # Gradient of b :
        self.dL_db = sum(dL_dZ, axis=0, keepdims=True)  # dL/db (for all samples) = dL/dZ * dZ/db where dZ/db=1
        self.b.grad += self.dL_db  # Accumulating on the W gradient
        # Computing dZn/dAn-1 : (ex : dZ2/dA1)
        dZ_dA = self.W.data.T
        dL_dA = dot(dL_dZ, dZ_dA)  # dL/dA = (dL/dZ)*(dZ_dA)
        return dL_dA

    def getParams(self):  # It is to provide access to the parameters of a specific layer (here is the linear layer)
        return [self.W, self.b]  # it is list of two arrays as : [array([[..],...,[...]]),array([[...]])]
