from numpy import exp, log, max, sum


class Multinomial:  # deal with as a loss function
    def forward(self, YHat, Y):
        exponent = exp(YHat - max(YHat, axis=1, keepdims=True))
        self.multinomial = exponent / sum(exponent, axis=1, keepdims=True)
        self.Y = Y
        loss = - log(self.multinomial[range(len(Y)), Y])  # loss is of shape (1, Y[0])
        return loss.mean()

    def backward(self, YHat):
        dL_dYHat = self.multinomial
        dL_dYHat[range(len(self.Y)), self.Y] -= 1.0
        dL_dYHat /= len(self.Y)
        return dL_dYHat  # shape (number of examples , number of nodes of output layer)
