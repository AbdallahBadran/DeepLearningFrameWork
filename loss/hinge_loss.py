from .loss import Loss
from numpy import where, sum


class HingeLoss(Loss):
    def forward(self, y_hat, y):
        self.Y = y
        product = y * y_hat
        loss = where(product > 1, 0, 1 - product)
        total = sum(loss, axis=0)
        return total / len(y)

    def backward(self, y_hat):
        product = self.Y * y_hat
        return where(product > 1, 0, - self.Y)
