from numpy import ndarray, float32


class DataAndGrad:
    def __init__(self, shape):
        # Defining the shape of weights and biases
        self.data = ndarray(shape, float32)
        # Defining the shape of the gradient of weights and biases
        self.grad = ndarray(shape, float32)

    # In[7]:
