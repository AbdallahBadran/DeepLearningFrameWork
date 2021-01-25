from .activation import Activation
from numpy import where


class HardTanh(Activation):
    def forward(self, z):
        self.Z = z
        temp = where(z < -1, -1, z)
        return where(temp > 1, 1, temp)

    def backward(self, grad):
        hard_tanh_grad = where(self.Z**2 < 1, 1, 0)
        return hard_tanh_grad * grad
