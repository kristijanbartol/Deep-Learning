import numpy as np
from copy import deepcopy
import sys

from model import RNN
from dataset import DataLoader

conversations_path = 'data/selected_conversations.txt'
generated_text_path = 'data/generated_text.txt'

lr = 1e-2
lr_coef = 2                 # coefficient for decrease current learning rate
max_diverging_steps = 5     # diverging steps limit after which learning rate decreases by lr_coef

batch_size = 1
max_epochs = 20
sequence_length = 20
hidden_size = 100


def batch_to_oh_input(x, vsize):
    input_x = np.zeros((x.shape[0], x.shape[1], vsize))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            input_x[i][j][x[i][j]] = 1
    return input_x


def seq_to_oh_input(idx, vsize, seq_length=1):
    input_x = np.zeros((1, seq_length, vsize))
    for i in range(seq_length):
        input_x[0][i][idx] = 1
    return input_x


def oh_to_char(oh_letter, data_loader):
    idx = [i for i, e in enumerate(oh_letter) if e != 0][0]
    return data_loader.id2char[idx]


def sample(rnn, data_loader, seed, n_sample):
    h0_sample = np.zeros((1, hidden_size))  # h0_shape -> (batch_size x hidden_size)
    old_sequence_length = rnn.sequence_length
    rnn.sequence_length = 1     # temporary hack to generate sequences of length 1

    # warming up the network with seed sequence
    context = deepcopy(h0_sample)
    for c in seed:
        input_x = seq_to_oh_input(data_loader.encode(c), len(data_loader.sorted_chars))
        context = rnn.refresh_context(input_x, context)

    current_letter_oh = rnn.generate_next(context)
    sample = []
    for _ in range(n_sample):
        context = rnn.refresh_context(current_letter_oh.reshape(1, 1, -1), context)
        current_letter_oh = rnn.generate_next(context)
        sample.append(oh_to_char(current_letter_oh, data_loader))

    rnn.sequence_length = old_sequence_length

    return ''.join(sample)


def generate_to_file(i, fgen, sample):
    fgen.write('=== {}. batch ===\n{}\n'.format(i, sample))


def run_language_model(data_loader, max_epochs, batch_size,
                       hidden_size=100, sequence_length=30, learning_rate=1e-2, sample_every=100):
    vocab_size = len(data_loader.sorted_chars)
    rnn = RNN(hidden_size, sequence_length, vocab_size, learning_rate)  # initialize the recurrent network

    epoch = 0
    batch = 0

    h0 = np.zeros((batch_size, hidden_size))

    sum_loss = 0
    best_avg_loss = sys.float_info.max
    diverging_steps = 0                     # counting number of steps where average loss started to increase
    fgen = open(generated_text_path, 'w')   # open file for generating output samples

    while epoch < max_epochs:
        e, x, y = data_loader.next_minibatch()

        if e:
            epoch += 1
            h0 = np.zeros((batch_size, hidden_size))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = batch_to_oh_input(x, vocab_size), batch_to_oh_input(y, vocab_size)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = rnn.step(h0, x_oh, y_oh)
        sum_loss += loss

        if batch % sample_every == 0:
            # run sampling (2.2)
            avg_loss = sum_loss / (batch + 1)
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                diverging_steps = 0
            else:
                diverging_steps += 1
                if diverging_steps > max_diverging_steps:
                    diverging_steps = 0
                    if learning_rate < 5e-4:
                        print('Learning rate at the lowest point -- no more decreasing...')
                    else:
                        learning_rate /= lr_coef

            best_avg_loss = sum_loss / (batch + 1)
            print('epoch: {} | batch: {} | loss: {} | avg_loss: {} | learning_rate {}'
                  .format(epoch + 1, batch + 1, loss, sum_loss / (batch + 1), learning_rate))
            generate_to_file(batch, fgen, sample(rnn, data_loader, seed='HAN:\nIs that good or bad?\n\n', n_sample=300))

        batch += 1


if __name__ == '__main__':
    data_loader = DataLoader(batch_size=batch_size, sequence_length=sequence_length)
    data_loader.preprocess(conversations_path)
    data_loader.create_minibatches()
    run_language_model(data_loader, max_epochs, batch_size, sequence_length=sequence_length, learning_rate=lr)

    # Testing
    # rnn = RNN(hidden_size, 1, len(data_loader.sorted_chars), 1e-1)
    # print(sample(rnn, data_loader, seed='HAN:\nIs that good or bad?\n\n', n_sample=300))
