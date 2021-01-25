from .activation import Activation


class Identity(Activation):
    def forward(self, z):
        return z

    def backward(self, grad):
        return grad  # The gradient of the identity function = 1
