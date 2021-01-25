from pandas import crosstab, Series
from numpy import diag, sum


class EvaluationMetrics:
    def accuracy(self, Y, ypred):
        accur = (sum(ypred == Y) / len(Y)) * 100
        return accur

    def confusionMatrix(self, Y, ypred):
        ypred = ypred
        Y = Y
        Y = Series(Y, name='Actual')
        ypred = Series(ypred, name='Predicted')
        df_confusion = crosstab(Y, ypred)
        return df_confusion

    def precision(self, cm):
        return diag(cm) / sum(cm, axis=0)

    def recall(self, cm):
        return diag(cm) / sum(cm, axis=1)

    def f1Score(self, cm):
        R = self.recall(cm)
        P = self.precision(cm)
        return (2 * P * R) / (P + R)