import torch.nn as nn
import torch.nn.functional as F
import torch


class HierarchicalAttentionNet(nn.Module):
    def __init__(self, args):
        super(HierarchicalAttentionNet, self).__init__()
        self.batch_size = args.batch_size
        self.hidden_size = args.hidden_size
        self.num_class = args.num_class
        self.word_attention_layer = WordAttentionNet(embedding_vectors=args.embedding_vectors,
                                                     hidden_size=self.hidden_size,
                                                     retrain_emb=args.retrain_emb)
        self.sentence_attention_layer = SentenceAttentionNet(hidden_size=self.hidden_size, num_class=self.num_class)

    def forward(self, x, **kwargs):
        # x is for batch 16 [16, 62, 271]: batch, tokens, sentences
        x = x.permute(1, 2, 0)  # becomes [62, 271, 16]: sentences, tokens, batch

        num_sentences = x.size(0)
        word_attentions = None

        for i in range(num_sentences):
            word_attn = self.word_attention_layer(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)

        output = self.sentence_attention_layer(word_attentions)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.002)


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, context_weights):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=True)
        self.context_weights = context_weights

    def forward(self, hidden_state):
        u_it = torch.tanh(self.linear(hidden_state))
        a_it = torch.matmul(u_it.transpose(1, 0), self.context_weights)
        a_it = torch.sigmoid(a_it)
        a_it = a_it.squeeze(dim=2)  # [8,126,1] becomes [8,126]
        s_i = torch.mul(hidden_state.permute(2, 0, 1), a_it.transpose(1, 0))
        s_i = torch.sum(s_i, dim=1).transpose(1, 0).unsqueeze(0)
        return s_i


class WordAttentionNet(nn.Module):

    def __init__(self, embedding_vectors, hidden_size, retrain_emb):
        super(WordAttentionNet, self).__init__()

        # Embedding layer
        vocab_size, embedding_size = embedding_vectors.shape
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.embedding_layer.weight.data.copy_(embedding_vectors)
        self.embedding_layer.weight.requires_grad = retrain_emb

        # bidirectional GRU layer
        self.gru_layer = nn.GRU(embedding_size, hidden_size, batch_first=True, bidirectional=True)

        # defining weights, since bidirectional we multiply the inputs by 2
        self.context_weights = nn.Parameter(torch.randn(2 * hidden_size, 1))
        #         self.context_weights.data.uniform_(-0.25, 0.25)

        # attention layer
        self.attention_layer = AttentionLayer(hidden_size, self.context_weights)

    def forward(self, x):
        embeddings = self.embedding_layer(x)
        output, hidden_state = self.gru_layer(embeddings)
        weights = self.attention_layer(output)
        return weights


class SentenceAttentionNet(nn.Module):
    def __init__(self, hidden_size, num_class):
        super(SentenceAttentionNet, self).__init__()

        # sentence and context weights
        self.context_weights = nn.Parameter(torch.randn(2 * hidden_size, 1))
        #         self.context_weights.data.uniform_(-0.25, 0.25)

        # bidirectional layer
        self.gru_layer = nn.GRU(2 * hidden_size, hidden_size, batch_first=True, bidirectional=True)

        # attention layer
        self.attention_layer = AttentionLayer(hidden_size, self.context_weights)

        # fc layer
        self.fc = nn.Linear(2 * hidden_size, num_class)

    def forward(self, hidden_state):
        output, hidden_state = self.gru_layer(hidden_state)
        weights = self.attention_layer(output)
        output = self.fc(weights.squeeze())
        return output
