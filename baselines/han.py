import torch
import torch.nn as nn
import math


# edited the code from https://github.com/castorini/hedwig

class WordLevelRNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        vectors = args.vectors
        word_num_hidden = args.word_num_hidden

        words_num, words_dim = vectors.shape
        self.embed = nn.Embedding.from_pretrained(vectors, freeze=False)
        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))

        stdv = 1. / math.sqrt(self.word_context_weights.size(0))
        self.word_context_weights.data.normal_(mean=0, std=stdv)

        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.soft_word = nn.Softmax(dim=-1)

    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        x = self.embed(x)
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x


class SentLevelRNN(nn.Module):

    def __init__(self, args):
        super().__init__()

        sentence_num_hidden = args.sentence_num_hidden
        word_num_hidden = args.word_num_hidden
        target_class = args.target_class

        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))

        stdv = 1. / math.sqrt(self.sentence_context_weights.size(0))
        self.sentence_context_weights.data.normal_(mean=0, std=stdv)

        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True, batch_first=True)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden, target_class)
        self.soft_sent = nn.Softmax(dim=-1)

    def forward(self, x):
        sentence_h, _ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1, 0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x


class HAN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.word_attention_rnn = WordLevelRNN(args)
        self.sentence_attention_rnn = SentLevelRNN(args)

    def forward(self, x, **kwargs):
        x = x.permute(1, 2, 0)  # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        return self.sentence_attention_rnn(word_attentions).squeeze(1)
