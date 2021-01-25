from matplotlib.pyplot import plot, xlabel, ylabel
from numpy import apply_along_axis, argmax, ceil, sum, zeros_like
from models.evaluation_metrics import EvaluationMetrics
from .linear import Linear
import pickle


class Model:
    def __init__(self):
        self.layers = []  # list of instances of all layers of the network
        self.parameters = []  # list of parameters of all linear layers of the network

    # Adding the passed layer to our network
    def add(self, layer):
        self.layers.append(layer)  # The actual addition of the instant to the list of instances
        if isinstance(layer, Linear):
            self.parameters += layer.getParams()  # Calling the getParams() function of specific passed layer
        # It is to concatenate all parameters of all layers in one list of parameters

    def saveModel(self):
        with open("model.pickle", "wb") as f:
            pickle.dump(self, f)

    def loadModel(self):
        pickleIn = open("model.pickle", "rb")
        modelLoaded = pickle.load(pickleIn)
        return modelLoaded

    # BatchPredictions = [] list of all outputs
    BatchPredictions = []

    def train(self, X, Y, batchSize, epochs, optimizer, lossFn, mode):
        # mode = 0: first time training, mode = 1: train using previous results
        if mode == 1:
            # load model
            var = self.loadModel()
            self.layers = var.layers
            self.parameters = var.parameters

        # print(var)
        # print(var.layers[0].W.data)
        # print('\n\n')
        # print(var.parameters[0].data)
        # print('\n\n')
        # print(var.layers[0].W.grad)
        # print(var.parameters[0].grad)
        # # print(var.layers[1])
        # # print(var.layers[2])
        # print(var.parameters[1])
        # print(var.parameters[2])
        # print(var.parameters[3])


        # Calculating the number of batches :
        numBatches = int(ceil(X.shape[0] / batchSize))
        # looping over the overall dataset one by one :
        LossHistory = []
        for epoch in range(epochs):
            # BatchPredictions = [] list of all outputs
            BatchPredictions = []
            # Initialize a counter over all batches :
            BatchesCounter = 0
            # For accumulating the batch losses :
            batchLossAcc = 0
            # Looping over all batches one by one :
            while BatchesCounter < numBatches:
                # Generating the batch data :
                # X [firstRowInBatch : LastRowInBatch]
                XBatch = X[BatchesCounter * batchSize: (BatchesCounter + 1) * batchSize]
                YBatch = Y[BatchesCounter * batchSize: (BatchesCounter + 1) * batchSize]

                # Zeroing Gradients before each batch :
                for p in self.parameters:
                    p.grad = zeros_like(p.grad)

                # Forward propagation :
                # The for loop recursively computes the forward propagation , the output of a
                # layer is put again as X to be input to the next layer :
                for layer in self.layers:
                    XBatch = layer.forward(XBatch)
                    # The final XBatch is (batch_size, number Of Nodes Of LastLayer)

                # BatchPredictions = A2.append [XBatch]
                BatchPredictions.extend(XBatch.tolist())  # output predictions values

                # Calculating the batch loss :
                # The total loss of the current batch :
                batchLoss = lossFn.forward(XBatch, YBatch)
                # print ("\nLoss of batch " ,BatchesCounter,"is : ", batchLoss, "\n")
                # Accumulating the batch loss :
                batchLossAcc += batchLoss

                # Backward propagation :
                grad = lossFn.backward(XBatch)  # This grad is of shape (batchSize, number Of Nodes Of LastLayer)
                for i, layer in enumerate(self.layers[::-1]):  # [::-1] is to start from the last layer till the first layer
                    grad = layer.backward(grad)  # model.layers = [Z1, A1, Z2, A2]    [W1, B1, W2, B2]
                    if isinstance(layer, Linear):
                        self.parameters[len(self.layers) - i - 1].grad = layer.W.grad
                        self.parameters[len(self.layers) - i].grad = layer.b.grad

                # for layer in self.layers[::-1]:  # [::-1] is to start from the last layer till the first layer
                #     grad = layer.backward(grad)  # model.layers = [Z1, A1, Z2, A2]

                # Updating parameters
                optimizer.parameters = self.parameters
                optimizer.update()
                self.parameters = optimizer.parameters

                # Increment the number of the batch for the next batch
                BatchesCounter += 1

            # function to get maximum prediction output of each (example)row  in each iteration

            def function(x):
                return argmax(x)

            ypred = apply_along_axis(function, 1, BatchPredictions)

            # Calculating the epoch loss :
            epochLoss = batchLossAcc / numBatches
            print("The loss of epoch ", epoch, "is : ", epochLoss)

            LossHistory.append(epochLoss)

            # accuracy
            Metric = EvaluationMetrics()
            acc = Metric.accuracy(Y, ypred)
            print("\n The accuracy of epoch ", epoch, "is : ", acc, "%\n")
            conf = Metric.confusionMatrix(Y, ypred)
            print("\n The confusion matrix of epoch ", epoch, "is : \n", conf, "\n")
            precision = Metric.precision(conf)
            print("\n The precision of epoch ", epoch, "is : ", precision, "\n")
            recall = Metric.recall(conf)
            print("\n The recall of epoch ", epoch, "is : ", recall, "\n")
            f1Score = Metric.f1Score(conf)
            print("\n The f1Score of epoch ", epoch, "is : ", f1Score, "\n")
            print("------------------------------------------------------------\n")

            # print(self.parameters[1].data)
            # print("grad")
            # print(self.parameters[1].grad)

        # end of for loop
        epochsList = range(epochs)
        plot(epochsList, LossHistory)
        xlabel('Epochs')
        ylabel('Epoch Loss')

        self.saveModel()

    def evaluate(self, data, labels):
        def function(x):
            return argmax(x)

        X = data
        for layer in self.layers:
            X = layer.forward(X)

        ypred = apply_along_axis(function, 1, X)
        accuracy = (sum(ypred == labels) / len(labels)) * 100
        print('Validation Accuracy: ' + str(accuracy) + '\n')
