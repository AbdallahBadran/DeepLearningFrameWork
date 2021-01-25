from .activation import Activation
from numpy import tanh


class Tanh(Activation):
    def forward(self, z):
        self.Z = z
        return tanh(z)

    def backward(self, grad):
        tanh_grad = 1 - tanh(self.Z)**2  # The gradient of the tanh itself
        return tanh_grad * grad  # The accumulated gradient
