from .loss import Loss
from numpy import exp, log, sum


class LogIdentity(Loss):
    def forward(self, y_hat, y):
        self.Y = y
        loss = log(1 + exp(-y * y_hat))
        total = sum(loss, axis=0)
        return total / len(y)

    def backward(self, y_hat):
        return -self.Y / (exp(self.Y * y_hat) + 1)
