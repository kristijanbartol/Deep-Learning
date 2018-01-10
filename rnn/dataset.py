import numpy as np

conversations_path = 'data/selected_conversations.txt'


class DataLoader:

    def __init__(self, batch_size=10, sequence_length=5):
        self.sorted_chars = []
        self.sequence_length = sequence_length
        self.char2id = dict()
        self.id2char = dict()
        self.x = np.zeros(0)

        self.batch_size = batch_size
        self.num_batches = -1
        self.current_batch = 0
        self.minibatches = np.zeros(0)

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = bytes(f.read(), 'utf-8').decode('utf-8')

        char_freq = dict()
        # count and sort most frequent characters
        for c in data:
            if c not in char_freq:
                char_freq[c] = 1
            else:
                char_freq[c] += 1
        self.sorted_chars = [k for k in sorted(char_freq, key=char_freq.get, reverse=True)]

        # self.sorted_chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return ''.join(list(map(self.id2char.get, encoded_sequence)))

    def create_minibatches(self):
        if self.sequence_length * self.batch_size > self.x.shape[0]:
            print('sequence length * batch size > dataset size (number of batches is zero)')
        # Note: not all samples are included
        self.num_batches = len(self.x) // (self.sequence_length * self.batch_size)
        current_idx = 0
        self.minibatches = []
        for i in range(self.num_batches):
            minibatch = ([], [])        # (batch_x, batch_y)
            for j in range(self.batch_size):
                minibatch[0].append(self.x[current_idx:current_idx+self.sequence_length])
                current_idx += self.sequence_length
                # skip whole overflowing batch (even though it shouldn't happen)
                if current_idx + self.sequence_length >= self.x.shape[0]:
                    break
                minibatch[1].append(self.x[current_idx:current_idx+self.sequence_length])
            self.minibatches.append(minibatch)
        self.minibatches = np.array(self.minibatches)

    def next_minibatch(self):
        new_epoch = False
        if self.current_batch == self.num_batches:
            new_epoch = True
            self.current_batch = 0
        batch_x, batch_y = self.minibatches[self.current_batch]
        self.current_batch += 1
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        return new_epoch, np.array(batch_x), np.array(batch_y)


if __name__ == '__main__':
    data_loader = DataLoader(batch_size=4, sequence_length=5)
    data_loader.preprocess(conversations_path)
    print(data_loader.encode('banana'))
    print(data_loader.decode(np.array([28, 5, 6, 5, 6, 5])))
    data_loader.create_minibatches()
    print('first batch: {}'.format(data_loader.next_minibatch()[1:]))
    print('second batch: {}'.format(data_loader.next_minibatch()[1:]))
    print(data_loader.minibatches.shape)
    print(data_loader.next_minibatch()[1].shape)
