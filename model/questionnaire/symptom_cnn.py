import torch
from torch import nn
from torch.nn import functional as F

class SymptomCNN(nn.Module):
    def __init__(self, embedding_dim=768, n_filters = 1, filter_sizes = (2, 3, 4, 5, 6), output_dim = 2, dropout=0.2, pool='max'):
        # =================================================
        # ARGUMENTS
        # - embedding_dim (int): embedding dimension of bert output (default: 768)
        # - n_filters (int): the number of filters
        # - filter_size (list or tuple)
        # - output_dim (int): output dimension after fc layer
        # - pool (str): pooling method
        #               supported methods are {'max', 'k-max', 'mix', 'avg'}
        # =================================================

        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.output_dim = output_dim
        self.dropout_prob = dropout
        self.pool = pool

        # CNN Layers
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.n_filters, kernel_size=(fs, self.embedding_dim)) for fs in self.filter_sizes]
        )

        # Fully Connected Layer
        if self.pool == 'k-max':
            self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters * 5, self.output_dim)
        elif self.pool == 'mix':
            self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters * 10, self.output_dim)
        else:
            self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters, self.output_dim)

        self.dropout = nn.Dropout(self.dropout_prob)

        # initialize weight
        for conv_layer in self.convs:
            self.init_weights(conv_layer)
        self.init_weights(self.fc)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.1)

    def forward(self, bert_encoded_output):
        # ======================================
        # INPUT
        # - bert_encoded_output: 'last_hidden_layer' of bert output
        #                       (batch_size, seq_len = MAX_LEN, hidden_size = embedding_dim)
        #
        # OUTPUT
        # - output: Probability vector for presence of the symptom
        #           (batch_size, output_dim)
        # - concat: hidden layer of symptom model
        #           for max pool, (b, n_filters*len(filter_sizes))
        # ======================================
        # bert_encoded_output (batch_size, seq_len=MAX_LEN, hid_size=embedding_dim)
        bert_encoded_output = bert_encoded_output.unsqueeze(1)  # (batch_size, 1, seq_len, hidden_size)

        # CNN (kernel size = (2, 3, 4, 5, 6)) and relu)
        conved = [F.relu(conv(bert_encoded_output)).squeeze(3) for conv in self.convs]  # [(b, n_filters, H) * 5]
                                                                                        # H = seq_len - kernel_size(fs) + 1
        # Pooling Layer
        if self.pool == 'max':
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [(b, n_filters) * len(filter_sizes)]
        elif self.pool == "k-max":
            batch_size = bert_encoded_output.size(0)
            pooled = [conv.topk(5, dim=2)[0].view(batch_size, -1) for conv in conved]
        elif self.pool == "mix":
            batch_size = bert_encoded_output.size(0)
            pooled = [torch.cat([conv.topk(5, dim=2)[0].view(batch_size, -1),
                                 conv.topk(5, dim=2, largest=False)[0].view(batch_size, -1)], dim=1) for conv in conved]
        elif self.pool == "avg":
            pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        else:
            raise ValueError("This kernel is currently not supported.")

        concat = torch.cat(pooled, dim=1)   # (b, n_filters*len(filter_sizes))
        concat = self.dropout(concat)
        output = self.fc(concat)   # (b, n_filters*len(filter_sizes)) -> (b, output_dim)

        if self.output_dim == 1:
            output = F.sigmoid(output)
        else:
            output = F.softmax(output, dim=1)

        return output, concat
