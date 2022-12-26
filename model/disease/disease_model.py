import torch, sys
import torch
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
from questionnaire.questionnaire_model import QuestionnaireModel

class DiseaseModel(nn.Module):
    def __init__(self, hidden_dim=5, n_filters=50, filter_sizes=(2, 3, 4, 5, 6), output_dim=1, dropout=0.2, num_symptom=None, pool='k-max', k=5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.output_dim = 1
        self.dropout_p = dropout
        self.pool = pool
        self.output_dim = output_dim
        self.max_k = []
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=self.n_filters, kernel_size=(fs, self.hidden_dim)) for fs in self.filter_sizes]
        )

        # Fully Connected Layer
        if self.pool == 'k-max':
            for fs in filter_sizes:
                H = num_symptom - fs + 1
                self.max_k.append(k) if k <= H else self.max_k.append(H)
            total_k = sum(self.max_k)
            self.fc = nn.Linear(total_k * self.n_filters, self.output_dim)

        elif self.pool == 'mix':
            for fs in filter_sizes:
                H = num_symptom - fs + 1
                self.max_k.append(k) if k <= H else self.max_k.append(H)
            total_k = sum(self.max_k)
            self.fc = nn.Linear(total_k * 2 * self.n_filters, self.output_dim)
        else:
            self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters, self.output_dim)

        self.dropout = nn.Dropout(self.dropout_p)
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

    def forward(self, question_model_output):
        # ====================================
        #   INPUT
        #   - question_model_output: (BATCH_SIZE, NUM_SYMPTOM, HIDDEN_DIM)
        #
        #   OUTPUT
        #   - output: Probability vector for presence of the symptom
        #               (BATCH_SIZE, output_dim)
        #   - concat: hidden layer of symptom model
        #               for max pool, (b, max_k * n_filters * len(filter_sizes))
        # ====================================
        question_model_output = question_model_output.unsqueeze(1)  # (BATCH_SIZE, 1, NUM_SYMPTOM, HIDDEN_DIM)
        conved = [F.relu(conv(question_model_output)).squeeze(3) for conv in self.convs]  # [(b, out_channel (n_filters), H) * len(filter_sizes)]
                                                                                        # H = NUM_SYMPTOM - kernel_size(fs) + 1
        if self.pool == 'max':
            pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        elif self.pool == 'k-max':
            batch_size = question_model_output.size(0)
            pooled = [conv.topk(tk, dim=2)[0].view(batch_size, -1) for conv, tk in zip(conved, self.max_k)] # [(b, tk * out_channels) * len(filter_sizes)]
        elif self.pool == 'mix':
            batch_size = question_model_output.size(0)
            pooled = [torch.cat([conv.topk(tk, dim=2)[0].view(batch_size, -1),
                                 conv.topk(tk, dim=2, largest=False)[0].view(batch_size, -1)], dim=1) for conv, tk in zip(conved, self.max_k)]
        elif self.pool == 'avg':
            pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        else:
            raise ValueError('This pooling method is not supported.')

        concat = torch.cat(pooled, dim=1)  # (b,  total_k * n_filters)
        concat = self.dropout(concat)
        output = self.fc(concat)  # (b, max_k * n_filters*len(filter_sizes)) -> (b, output_dim)

        if self.output_dim == 1:
            output = self.sigmoid(output)
        else:
            output = self.softmax(output)

        return output, concat




if __name__ == '__main__':
    from train_question_model import get_args

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

    # Load Data
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    question_model = QuestionnaireModel(num_symptoms=9, filter_sizes=(2,))
    loss_fn = nn.BCELoss()

    device = torch.device("cuda")
    bert_model.cuda()
    question_model.cuda()

    optimizer = torch.optim.AdamW(
        question_model.parameters(),
        lr=args.lr,
        betas=args.betas,
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * (train_dataset.num_data / args.batch_size),
    )

    for epoch_i in range(args.epochs):
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

            symptom_scores, symptom_labels, symptom_hidden = question_model(bert_output, labels) # (b, num_symptom, 1), (b, num_symptom, 1), (b, 5)
            #preds = [[1] if prob.item() > 0.5 else [0] for prob in symptom_output]

            loss = loss_fn(symptom_scores.to(torch.float32), symptom_labels.to(torch.float32).to(device))
            total_loss += loss.item()
            loss_for_logging += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()


            break

        # epoch ends
        # print results
        print("EPOCH {}".format(epoch_i))
        print("Total train loss: {}".format(total_loss/len(train_dl)))

    import IPython;
    IPython.embed();
    exit(1)