from torch import cat
from torch import nn
from torch import zeros, rand
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class biGRU(nn.Module):
    def __init__(self, words_dim, hidden_size, weights_matrix, fine_tune, num_output=1, device='cpu'):
        super().__init__()
        self.hidden_size = hidden_size
        self.words = words_dim
        freeze = False if fine_tune else True
        self.lookup = nn.Embedding.from_pretrained(weights_matrix, freeze=freeze, padding_idx=0)
        self.gru = nn.GRU(input_size=words_dim, hidden_size=hidden_size, bidirectional=True,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_output)
        self.device = device

    def init_hidden(self, batch_size):
        hidden = Variable(rand(2, batch_size, self.hidden_size))
        return hidden

    def forward(self, inputs, inputs_len):
        # inputs are padded, and in the dimension of batch_size x sequence length
        embeddings = self.lookup(inputs)
        # TODO maybe we can add dropout, but we need to make sure that pads are not gone.

        # our inputs are padded so we use pack padded sequences not to show to rnn
        packed_seq = pack_padded_sequence(embeddings, inputs_len, batch_first=True, enforce_sorted=False)

        # gru out is the last layer of the GRU, for each t
        batch_size = inputs.shape[0]
        hidden = self.init_hidden(batch_size)

        packed_gru_out, hidden = self.gru(packed_seq,hidden)

        # restore to padded sequence
        gru_out, _ = pad_packed_sequence(packed_gru_out, batch_first=True)

        # gru_final = gru_out[:, -1, :]

        gru_final = cat([gru_out[:, -1, :self.hidden_size], gru_out[:, 0, self.hidden_size:]], 1)

        fc_output = self.fc(gru_final)

        return fc_output.flatten()
