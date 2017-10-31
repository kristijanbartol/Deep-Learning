import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# smece pythona
sys.path.append("lab0")

import data

param_niter = 100
param_delta = 0.25


# stabilni softmax
def stable_softmax(x):
    exp_x_shifted = np.exp(x - np.max(x))
    probs = exp_x_shifted / np.sum(exp_x_shifted)
    return probs


def logreg_train(X, Y_):
    C = max(Y_) + 1
    N, D = X.shape

    w = [np.random.randn(C) for _ in range(D)]  # D x C
    b = np.zeros(C).reshape(1, C)   # C x 1

    for i in range(param_niter):
        scores = np.dot(X, w) + b  # (N x D) x (D x C) + (1 x C) = N x C
        expscores_shifted = np.exp(scores - np.max(scores))  # N x C

        # nazivnik sofmaksa
        sumexp = np.sum(expscores_shifted, axis=0)  # N x 1

        # logaritmirane vjerojatnosti razreda
        probs = expscores_shifted / sumexp  # N x C
        logprobs = ...  # N x C

        # gubitak
        loss = ...  # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = ...  # N x C

        # gradijenti parametara
        grad_W = ...  # C x D (ili D x C)
        grad_b = ...  # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b
