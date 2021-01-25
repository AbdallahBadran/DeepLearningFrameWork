from .activation import Activation
from numpy import where


class Relu(Activation):
    def forward(self, z):
        self.Z = z
        return where(z < 0, 0, z)

    def backward(self, grad):
        return where(self.Z < 0, 0, grad)
