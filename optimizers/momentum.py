from .optimizer import Optimizer


class Momentum(Optimizer):
    def __init__(self, parameters, alpha, momentum=0.9):
        super().__init__(parameters, alpha)
        self.momentum = momentum  # BETA

        # elaccumulator howa elli ba accumulate feh fa lazem asafaro felbedia li kol elparameters
        for p in self.parameters:
            p.momentum_vector_parameters = 0  # V

    def update(self):
        for p in self.parameters:
            p.momentum_vector_parameters = (self.momentum * p.momentum_vector_parameters) - (self.alpha * p.grad)
            p.data = p.data + p.momentum_vector_parameters
