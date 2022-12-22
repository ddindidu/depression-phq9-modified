import torch, sys
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

sys.path.insert(0, './')
sys.path.insert(0, './../')
from bert_model import BertModelforBaseline, get_batch_bert_embedding
from dataset import DepressionDataset



class SymptomCNN(nn.Module):
    def __init__(self, embedding_dim=768, n_filters=1, filter_sizes=(2, 3, 4, 5, 6), output_dim=1, dropout=0.2, pool='max'):
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
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

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
            output = self.sigmoid(output)
        else:
            output = self.softmax(output)

        return output, concat


if __name__ == '__main__':
    from train import get_args

    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    bert_model = BertModelforBaseline(
        args,
        tokenizer=AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        ),
        bert_model=AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        ),
    )

    train_dataset = DepressionDataset(
        args=args,
        mode='train',
        tokenizer=tokenizer,
    )
    epochs = 3
    batch_size = 32
    # Load Data
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    symptom_model = SymptomCNN(filter_sizes=(2, 3))
    loss_fn = nn.BCELoss()

    device = torch.device("cuda")
    bert_model.cuda()
    symptom_model.cuda()

    optimizer = torch.optim.AdamW(
        symptom_model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )


    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=epochs * (train_dataset.num_data / batch_size),
    )

    for epoch_i in range(epochs):
        total_loss = 0
        loss_for_logging = 0
        for step, data in enumerate(tqdm(train_dl, desc='train', mininterval=0.01, leave=True), 0):
            inputs = {
                "input_ids": data['input_ids'].to(device),
                "attention_mask": data['attention_mask'].to(device),
                "token_type_ids": data['token_type_ids'].to(device),
            }
            labels = data['labels'].to(device) # (b)

            bert_output = get_batch_bert_embedding(bert_model, inputs, trainable=True)  # (batch, MAX_SEQUENCE_LEN, embedding_dim)

            symptom_output, symptom_hidden = symptom_model(bert_output) # (b, 1), (b, 5)
            preds = [[1] if prob.item() > 0.5 else [0] for prob in symptom_output]

            loss = loss_fn(symptom_output.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
            total_loss += loss.item()
            loss_for_logging += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()


            #break

        # epoch ends
        # print results
        print("EPOCH {}".format(epoch_i))
        print("Total train loss: {}".format(total_loss/len(train_dl)))

    import IPython;
    IPython.embed();
    exit(1)