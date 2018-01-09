import numpy as np

from model import RNN
from dataset import DataLoader

conversations_path = 'data/selected_conversations.txt'
batch_size = 1
max_epochs = 20


def to_oh(x, vsize):
    new_x = np.zeros((x.shape[0], vsize))
    for i in range(x.shape[0]):
        new_x[i][x[i]] = 1
    return new_x


def run_language_model(data_loader, max_epochs, batch_size, hidden_size=100, sequence_length=30, learning_rate=1e-1):
    vocab_size = len(data_loader.sorted_chars)
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)  # initialize the recurrent network

    current_epoch = 0
    batch = 0

    h0 = np.zeros((batch_size, hidden_size))

    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = data_loader.next_minibatch()

        if e:
            current_epoch += 1
            h0 = np.zeros((batch_size, hidden_size))
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
        print('loss: {} | avg_loss: {}'.format(loss, average_loss))

        batch += 1


if __name__ == '__main__':
    data_loader = DataLoader(batch_size=batch_size)
    data_loader.preprocess(conversations_path)
    data_loader.create_minibatches()
    run_language_model(data_loader, max_epochs, batch_size)
