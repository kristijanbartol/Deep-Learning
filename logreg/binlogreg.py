import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# smece pythona
sys.path.append("lab0")

import data

param_niter = 100
param_delta = 0.25


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def output_binary(X, Y_, w, b):
    scores = np.dot(X, w) + b
    scores_shifted = scores - np.max(scores)
    #for i in range(X.shape[0]):
    #    print("{} x {} + {} = {}\n{}\n\n".format(X[i], w, b, np.dot(X[i], w) + b, sigmoid(np.dot(X[i], w) + b)))

    return [sigmoid(scores_shifted[i]) for i in range(X.shape[0])]


def binlogreg_train(X, Y_):
    N, D = X.shape

    w = np.random.randn(D).reshape(D, 1)
    b = np.array([0.]).reshape(1, 1)
    Y_ = Y_.reshape(N, 1)

    last_loss = sys.maxsize

    for i in range(param_niter):
        probs = np.array(output_binary(X, Y_, w, b)).reshape(N, 1)
        loss = np.sum(-np.log(np.abs(probs - Y_)))

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))
        #if loss > last_loss:
        #    print("iteration {}: loss {}".format(i, loss))
        #    print("method converged at iteration {}: loss {}".format(i, loss))
        #    break
        #else:
        #    last_loss = loss

        #print (X.shape, np.abs(probs-Y_).T.shape)

        grad_w = 1 / N * np.dot(np.abs(probs - Y_).T, X)
        grad_b = 1 / N * np.sum(np.abs((probs - Y_)))

        #print(grad_w.shape)

        w += -param_delta * grad_w.T
        b += -param_delta * grad_b

    return w, b


def binlogreg_classify(X, w, b):
    scores = np.dot(X, w) + b
    scores_shifted = scores - np.max(scores)
    return [sigmoid(scores_shifted[i]) for i in range(X.shape[0])]


def binlogreg_decfun(w, b):
    def classify(X):
      return binlogreg_classify(X, w, b)
    return classify


if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss(2, 20)

    # train the model
    w, b = binlogreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = np.array(binlogreg_classify(X, w, b))
    Y = np.array([1 if p > 0.5 else 0 for p in probs])

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[np.argsort(probs)])
    print(accuracy, recall, precision, AP)

    # graph the decision surface
    decfun = binlogreg_decfun(w, b)
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    #data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
