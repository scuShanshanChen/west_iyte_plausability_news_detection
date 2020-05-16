import torch
from torch import nn
from torch.utils.data import DataLoader


class biGRU(nn.Module):
    def __init__(self, words_dim, hidden_size, weights_matrix, num_output=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.words = words_dim
        self.lookup = nn.Embedding.from_pretrained(weights_matrix, freeze=False, padding_idx=0)
        self.gru = nn.GRU(input_size=words_dim, hidden_size=hidden_size, bidirectional=True,
                          batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_output)

        # for name, params in self.gru.named_parameters():
        #     # weight: Orthogonal Initialization
        #     if 'weight' in name:
        #         nn.init.orthogonal_(params)
        #     # lstm forget gate bias init with 1.0
        #     if 'bias' in name:
        #         b_i, b_f, b_c, b_o = params.chunk(4, 0)
        #         nn.init.ones_(b_f)

    def forward(self, inputs):
        # inputs are padded, and in the dimension of batch_size x sequence length
        embeddings = self.lookup(inputs)
        # TODO maybe we can add dropout, but we need to make sure that pads are not gone.
        # gru out is the last layer of the GRU, for each t
        gru_out, hidden = self.gru(embeddings)

        # take the output of last sequence
        # the bellow commented out method actually didn't work, I don't know why
        # gru_final = gru_out[:, -1, :]
        gru_final = torch.cat([gru_out[:, -1, :self.hidden_size], gru_out[:, 0, self.hidden_size:]], 1)

        fc_output = self.fc(gru_final)

        return fc_output.flatten()
