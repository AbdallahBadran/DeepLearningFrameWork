from .loss import Loss
from numpy import reshape, square, sum


class MeanSquared(Loss):
    def forward(self, y_hat, y):
        self.Y = y
        self.Y = reshape(self.Y, (len(self.Y), -1))
        sqr = square(self.Y - y_hat)
        total = sum(sqr, axis=0)
        mse = 1 / (2 * len(self.Y)) * total  # list of mse of each node
        mean = mse.mean()  # The mean of all mse of the nodes
        return mean

    def backward(self, y_hat):
        return y_hat - self.Y  # dL/dA2
