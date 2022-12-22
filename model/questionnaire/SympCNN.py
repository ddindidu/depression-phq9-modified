import torch
from torch import nn
from torch.nn import functional as F

class SympCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim, pool='max'):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim)) for fs in filter_sizes]
        )


    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.1)


    def forward(self, bert_encoded_output):
        # bert_encoded_output (batch_size, seq_len, hid_size=768)
