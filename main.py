from pandas import read_csv
# from numpy import reshape
# import io

# from activation.hard_tanh import HardTanh
# from activation.identity import Identity
from activation.relu import Relu
# from activation.sigmoid import Sigmoid
# from activation.sign import Sign
# from activation.tanh import Tanh

# from loss.hinge_loss import HingeLoss
# from loss.log_identity import LogIdentity
# from loss.log_sigmoid import LogSigmoid
# from loss.mean_squared import MeanSquared
# from loss.smooth_loss import SmoothLoss

from models.linear import Linear
from models.model import Model
from models.multinomial import Multinomial

# from optimizers.ada_grad import AdaGrad
# from optimizers.momentum import Momentum
# from optimizers.rms_prop import RMSProp
from optimizers.stochastic_gradient_descent import SGD


def loadMnist(validationRatio):
    # ReadData
    # df = read_csv(io.StringIO(uploaded['train.csv'].decode('utf-8')))
    df = read_csv('./train.csv')
    # shuffle data
    shuffuled_df = df.sample(frac=1)
    X = shuffuled_df.drop(labels=['label'], axis=1).values
    # flattenng
    X = X/255
    Y = shuffuled_df.label.values  # output (list)
    testSetSize = int(len(X) * validationRatio)
    X_test = X[:testSetSize]
    Y_test = Y[:testSetSize]
    X_train = X[testSetSize:]
    Y_train = Y[testSetSize:]
    return (X_train, Y_train), (X_test,Y_test)


model = Model()

prevNodes = 784  # of the input layer (layer0)
currNodes = 100  # of the layer1
Z1 = Linear(prevNodes, currNodes)
model.add(Z1)

A1 = Relu()
model.add(A1)

currNodes = 10
prevNodes = 100
Z2 = Linear(prevNodes, currNodes)
model.add(Z2)

optimizer = SGD(model.parameters, alpha=1.0)
# optimizer = MomentumOptimizer(model.parameters, alpha = 1.0, momentum = 0.9)
# optimizer = AdaGrad(model.parameters, alpha = 1.0, epsilon = 10 ** -10)
# optimizer = RMSProp(model.parameters, alpha = 1.0, decay_rate = 0.9, epsilon = 10 ** -10)

lossFun = Multinomial()
batchSize = 100
epochs = 5

(X_train, Y_train), (X_test, Y_test) = loadMnist(0.2)

choice = input('0. first time\n1. use previous\n')
if choice == '0':
    model.train(X_train, Y_train, batchSize, epochs, optimizer, lossFun, 0)
elif choice == '1':
    model.train(X_train, Y_train, batchSize, epochs, optimizer, lossFun, 1)

model.evaluate(X_test, Y_test)

del model
