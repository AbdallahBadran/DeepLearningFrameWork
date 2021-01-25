from .optimizer import Optimizer


class SGD(Optimizer):
    def update(self):
        for p in self.parameters:
            p.data = p.data - self.alpha * p.grad  # update parameters
