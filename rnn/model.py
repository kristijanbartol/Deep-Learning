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

    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (batch size x input dimension)
        # h_prev - previous hidden state (hidden size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current = np.tanh(np.dot(h_prev, W) + np.dot(x, U) + b.T)     # (batch_size x hidden_size)

        # return the new hidden state and a tuple of values needed for the backward step
        return h_current, (h_current, h_prev, x)

    def _rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (batch size x sequence length x vocab size)
        # h0 - initial hidden state (batch size x hidden size)
        # U - input projection matrix (vocab size x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h = []
        h_current = copy.deepcopy(h0)
        cache = []
        x_swapped = np.swapaxes(x, 0, 1)
        for i in range(self.sequence_length):
            h_current, cache_ = self.rnn_step_forward(x_swapped[i], h_current, U, W, b)
            h.append(h_current)
            cache.append(cache_)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h, cache

    def refresh_context(self, x, h0):
        return self._rnn_forward(x, h0, self.U, self.W, self.b)[0][0]

    @staticmethod
    def rnn_step_backward(grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass
        # cache[0] <- h_t   (1 x hidden size)
        # cache[1] <- h_t-1
        # cache[2] <- x_t   (batch size x vocab size)
        # a <- argument of tanh

        da = grad_next * (1 - np.square(cache[0]))      # (batch_size x hidden_size)
        dh_prev = grad_next                             # (batch_size x hidden_size)
        dU = np.dot(cache[2].T, da)                     # (vocab_size x hidden_size)
        dW = np.dot(cache[1].T, da)                     # (hidden_size x hidden_size)
        db = da.T                                       # (hidden_size x batch_size)

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dh_ = copy.deepcopy(dh)
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        for i in range(self.sequence_length - 1, 0, -1):
            dh_, dU_, dW_, db_ = RNN.rnn_step_backward(dh_, cache[i])
            dU += np.clip(dU_, -5, 5)
            dW += np.clip(dW_, -5, 5)
            db += np.sum(np.clip(db_, -5, 5), axis=1).reshape(-1, 1)

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU, dW, db

    @staticmethod
    def _output(h, V, c):
        # Calculate the output probabilities of the network

        # V - hidden_size x vocab_size
        # h - batch_size x hidden_size
        # c - vocab_size x 1

        return np.dot(h, V) + c.T

    def output(self, h):
        yhat = []
        print(h.shape)
        for i in range(len(h)):
            yhat.append(RNN._output(h[i], self.V, self.c))
        return np.array(yhat)

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

        o = RNN._output(np.swapaxes(h, 0, 1), V, c)           # V * h_t + c   (batch_size x sequence_size x vocab_size)
        yhat = softmax(o)                                     # softmax(o_t)  (batch_size x sequence_size x vocab_size)
        loss = -1 / y.size * np.sum(y * np.log(yhat))         # (scalar)
        do = 1 / yhat.shape[1] * np.sum(yhat - y, axis=1)     # dL_do = yhat - y  (batch_size x vocab_size)
        dh = np.dot(do, V.T) + 0                      # dL_dh = V.T * dL_do + dL+1_h_t  (1 x vocab_size)
        dV = np.dot(h[:][-1].T, do)                   # dL_dV = (yhat - y_t) * h_t.T    (hidden_size x vocab_size)
        dc = 1 / do.T.shape[1] * np.sum(do.T, axis=1).reshape(-1, 1)      # yhat - y_t  (vocab_size x 1)

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        return loss, dh, dV, dc

    def idx_to_oh(self, idx):
        oh_v = np.zeros(self.vocab_size)
        oh_v[idx] = 1
        return oh_v

    def generate_next(self, h):
        return self.idx_to_oh(np.argmax(softmax(self._output([h], self.V, self.c)[0])))

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

    def step(self, h0, x_oh, y_oh):
        h, cache = self._rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y_oh)
        dU, dW, db = self.rnn_backward(dh, cache)

        self.update(dU, dW, db, dV, dc, self.U, self.W, self.b, self.V, self.c,
                    self.memory_U, self.memory_W, self.memory_b, self.memory_V, self.memory_c)

        return loss, h[:][-1]
