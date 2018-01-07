import numpy as np
import copy


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


class RNN:

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.small_const = 1e-7

        self.U = np.random.normal(size=(vocab_size, hidden_size))
        self.W = np.random.normal(size=(hidden_size, hidden_size))
        self.b = np.random.normal(size=(hidden_size, 1))

        self.V = np.random.normal(size=(hidden_size, vocab_size))
        self.c = np.random.normal(size=(vocab_size, 1))

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    @staticmethod
    def rnn_step_forward(x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current = np.tanh(W * h_prev.T + U.T * x.T + b)

        # return the new hidden state and a tuple of values needed for the backward step
        return h_current, (h_current, h_prev, x)

    @staticmethod
    def rnn_forward(x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h = [copy.deepcopy(h0)]
        cache = []
        for i in range(x.shape[1]):     # sequence_length
            h_, cache_ = RNN.rnn_step_forward(x, h[-1], U, W, b)
            h.append(h_)
            cache.append(cache_)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h, cache

    @staticmethod
    def rnn_step_backward(grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        # cache[0] <- h_t
        # cache[1] <- h_t-1
        # cache[2] <- x_t

        da = grad_next * (1 - np.square(cache[0]))      # a <- argument of tanh
        dh_prev = grad_next
        dU = da * cache[2].T
        dW = da * cache[1].T
        db = da

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db

    @staticmethod
    def rnn_backward(dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dh_ = dh
        dU = []
        dW = []
        db = []
        for i in range(dh.cache[0] - 1, 0, -1):
            dh_, dU_, dW_, db_ = RNN.rnn_step_backward(dh_, cache[i])
            dU.append(np.clip(dU_, -5, 5))
            dW.append(np.clip(dW_, -5, 5))
            db.append(np.clip(db_, -5, 5))

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU, dW, db

    @staticmethod
    def output(h, V, c):
        # Calculate the output probabilities of the network

        # V - vocab_size x hidden_size
        # h - hidden_size x batch_size
        # c - vocab_size x 1

        return V * h + c

    @staticmethod
    def output_loss_and_grads(h, V, c, y):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each time step.
        #     the dimensionality of h is batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        # y - the true class distribution - a tensor of dimension
        #     batch_size x sequence_length x vocabulary size - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        #   y[batch_id][timestep] = np.zeros((vocabulary_size, 1))
        #   y[batch_id][timestep][batch_y[timestep]] = 1

        #     where y might be a list or a dictionary.

        o = RNN.output(h[:][-1], V, c)
        yhat = softmax(o)
        loss = y - yhat
        dh = V.T * loss
        dV = loss * h[:][-1].T
        dc = loss

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        return loss, dh, dV, dc

    def update(self, dU, dW, db, dV, dc, U, W, b, V, c, memory_U, memory_W, memory_b, memory_V, memory_c):
        # update memory matrices
        # perform the Adagrad update of parameters
        memory_U += np.square(dU)
        memory_W += np.square(dW)
        memory_b += np.square(db)
        memory_V += np.square(dV)
        memory_c += np.square(dc)

        U -= self.learning_rate / (self.small_const + np.sqrt(memory_U)) * dU
        W -= self.learning_rate / (self.small_const + np.sqrt(memory_W)) * dW
        b -= self.learning_rate / (self.small_const + np.sqrt(memory_b)) * db
        V -= self.learning_rate / (self.small_const + np.sqrt(memory_V)) * dV
        c -= self.learning_rate / (self.small_const + np.sqrt(memory_c)) * dc

        # return U, W, b, V, c

    def step(self, h0, x_oh, y_oh):
        h, cache = self.rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)

        # self.U, self.W, self.b, self.V, self.c = self.update(dU, dW, db, dV, dc, self.U, self.W, self.b, self.V, self.c,
        #                                                     self.memory_U, self.memory_W, self.memory_b, self.memory_V, self.memory_c)
        self.update(dU, dW, db, dV, dc, self.U, self.W, self.b, self.V, self.c,
                    self.memory_U, self.memory_W, self.memory_b, self.memory_V, self.memory_c)

        return loss, h[:][-1]
