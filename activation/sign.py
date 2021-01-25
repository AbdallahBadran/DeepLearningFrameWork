from .activation import Activation
from numpy import where, zeros_like


class Sign(Activation):
    def forward(self, z):
        return where(z < 0, -1, 1)

    def backward(self, grad):
        return zeros_like(grad)  # The gradient of the sign function = 0
