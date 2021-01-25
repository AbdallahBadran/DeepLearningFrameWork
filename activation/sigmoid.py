from .activation import Activation
from numpy import exp


class Sigmoid(Activation):
    def forward(self, z):
        self.Z = z
        return 1 / (1 + exp(- self.Z))

    def backward(self, grad):
        sigmoid_grad = (1 / (1 + exp(- self.Z))) * (1 - (1 / (1 + exp(- self.Z))))  # The gradient of the sigmoid itself
        return sigmoid_grad * grad  # The accumulated gradient
