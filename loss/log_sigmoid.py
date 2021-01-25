from .loss import Loss
from numpy import log, sum


class LogSigmoid(Loss):
    def forward(self, y_hat, y):
        self.Y = y
        loss = -log(abs(0.5 * y - 0.5 + y_hat))
        total = sum(loss, axis=0)
        return total / (2 * len(y))

    def backward(self, y_hat):
        return -2 / (self.Y - 1 + 2 * y_hat)
