import sys
import numpy as np
import matplotlib.pyplot as plt

# smece pythona
sys.path.append("lab0")

import data

param_niter = 100
param_delta = 0.25


def output(X, W, b):
    scores = np.dot(X, W) + b  # (N x D) x (D x C) + (1 x C) = N x C
    expscores_shifted = np.exp(scores - np.max(scores))  # N x C

    # nazivnik sofmaksa
    sumexp = np.sum(expscores_shifted, axis=0)  # N x 1

    # logaritmirane vjerojatnosti razreda
    probs = np.divide(expscores_shifted, sumexp)  # N x C
    logprobs = np.log(probs)  # N x C
    return probs, logprobs


def one_hot_vector(n, i):
    a = np.zeros(n)
    a[i] = 1
    return a


def to_one_hot_matrix(Y, C):
    return [one_hot_vector(C, i) for i in Y]


def logreg_train(X, Y_):
    C = max(Y_) + 1
    N, D = X.shape

    W = [np.random.randn(C) for _ in range(D)]  # D x C
    b = np.zeros(C).reshape(1, C)   # C x 1

    for i in range(param_niter):
        probs, logprobs = output(X, W, b)

        # gubitak
        loss = -np.sum(logprobs)  # scalar

        # dijagnostički ispis
        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # derivacije komponenata gubitka po mjerama
        dL_ds = probs - to_one_hot_matrix(Y_, C)  # N x C

        # gradijenti parametara
        grad_W = 1 / N * np.dot(dL_ds.T, X)  # C x D (ili D x C)
        grad_b = 1 / N * np.sum(dL_ds, axis=0)  # C x 1 (ili 1 x C)

        # poboljšani parametri
        W += -param_delta * grad_W
        b += -param_delta * grad_b

    return W, b


def logreg_classify(X, W, b):
    return output(X, W, b)


def logreg_decfun(w, b):
    def classify(X):
      return logreg_classify(X, w, b)
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gmm(5, 2, 100)

    # train the model
    W, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs, logprobs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, confusion_matrix, recall = data.eval_perf_multi(Y, Y_)
    print(accuracy, confusion_matrix, recall)

    data.graph_data(X, Y_, Y, special=[])

    decfun = logreg_decfun(W, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    #data.graph_surface(decfun, bbox, offset=0.5)

    plt.show()
