from .optimizer import Optimizer
from numpy import sqrt


class AdaGrad(Optimizer):
    def __init__(self, parameters, alpha, epsilon=10 ** -10):
        super().__init__(parameters, alpha)
        self.epsilon = epsilon

        for p in self.parameters:
            p.scale_parameters = 0

    def update(self):
        for p in self.parameters:
            p.gradient_parameters = p.grad

            p.scale_parameters += p.gradient_parameters * p.gradient_parameters

            p.divider_parameters = sqrt(p.scale_parameters + self.epsilon)

            p.data -= self.alpha * p.gradient_parameters / p.divider_parameters
