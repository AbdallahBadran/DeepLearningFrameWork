from .optimizer import Optimizer
from numpy import sqrt


class RMSProp(Optimizer):
    def __init__(self, parameters, alpha, decay_rate=0.9, epsilon=10 ** -10):
        super().__init__(parameters, alpha)
        self.decay_rate = decay_rate
        self.epsilon = epsilon

        for p in self.parameters:
            p.scale_parameters = 0

    def update(self):
        for p in self.parameters:
            p.gradient_parameters = p.grad

            p.scale_parameters += (1 - self.decay_rate) * p.gradient_parameter * p.gradient_parameters

            p.divider_parameters = sqrt(p.scale_parameters + self.epsilon)

            p.data -= self.alpha * p.gradient_parameters / p.divider_parameters
