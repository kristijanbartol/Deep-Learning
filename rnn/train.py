import numpy as np

from model import RNN


def sample(seed, n_sample):
    h0, seed_onehot, sample = None, None, None
    # inicijalizirati h0 na vektor nula
    # seed string pretvoriti u one-hot reprezentaciju ulaza

    return sample


def to_oh(x, vsize):
    new_x = np.zeros((x.shape[0], x.shape[1], vsize))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            new_x[i][j][x[i][j]] = 1
    return new_x


def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)  # initialize the recurrent network

    current_epoch = 0
    batch = 0

    h0 = np.zeros((hidden_size, 1))

    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((hidden_size, 1))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = to_oh(x, vocab_size), to_oh(y, vocab_size)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)

        if batch % sample_every == 0:
            # run sampling (2.2)
            pass
        batch += 1
