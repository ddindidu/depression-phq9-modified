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
from questionnaire.symptom_cnn import SymptomCNN

class QuestionnaireModel(nn.Module):
    def __init__(self,
                 num_symptoms,
                 embedding_dim=768,
                 n_filters=1,
                 filter_sizes=(2, 3, 4, 5, 6),
                 output_dim=1,
                 dropout=0.2,
                 pool='max'):
        super(QuestionnaireModel, self).__init__()
        self.num_symptoms = num_symptoms
        self.question_models = nn.ModuleList(
            [SymptomCNN(embedding_dim,
                        n_filters,
                        filter_sizes,
                        output_dim,
                        dropout,
                        pool) for _ in range(self.num_symptoms)]
        )

    '''
    def run_all_symptoms(self, bert_output):
        # ====================================
        #   INPUT
        #   - bert_output: (batch_size, MAX_SEQ_LEN, EMB_DIM)
        #
        #   OUTPUT
        #   - res_sym_prob: (BATCH_SIZE, NUM_SYMP, 1)
        #   - res_sym_hidden: (BATCH_SIZE, NUM_SYMP, n_filters * len(filter_sizes))
        # ====================================

        res_sym_prob, res_sym_hidden = [], []
        for sym_model in self.question_models:
            symptom_prob, symptom_hidden = sym_model(bert_output) # (b, 1), (b, n_filters * len(filter_sizes))
            res_sym_prob.append(symptom_prob)       # (num_symptoms, b, 1)
            res_sym_hidden.append(symptom_hidden)   # (num_symptoms, b, n_filters * len(filter_sizes))

        res_sym_prob = torch.stack(res_sym_prob).transpose(0, 1)    # (b, num_symp, 1)
        res_sym_hidden = torch.stack(res_sym_hidden).transpose(0, 1)  # (b, num_symp, n_fil * len(fs))

        return res_sym_prob, res_sym_hidden
    '''

    def forward(self, bert_output, labels):
        # ====================================
        #   INPUT
        #   - bert_output: (batch_size, MAX_SEQ_LEN, EMB_DIM)
        #   - labels (list of int): list of symptom number (BATCH_SIZE)
        #
        #   OUTPUT
        #   - symptom_scores: (BATCH_SIZE, NUM_SYMP, 1)
        #   - sym_labels: (BATCH_SIZE, NUM_SYMP, 1)
        #   - symptom_vectors: hidden vectors for symptoms(BATCH_SIZE, NUM_SYMP, n_filters * len(filter_sizes))
        # ====================================

        batch_size = bert_output.size(0)
        sym_labels = torch.zeros(batch_size, self.num_symptoms)
        for batch_ind, symp_no in enumerate(labels):
            # if symptom number is 2,
            # sym_labels becomes [0, 0, 1, 0, ..., 0]
            sym_labels[batch_ind, symp_no] = 1
        sym_labels = sym_labels.unsqueeze(-1)   # (b, num_symp, 1)

        res_sym_prob, res_sym_hidden = [], []
        for sym_model in self.question_models:
            symptom_prob, symptom_hidden = sym_model(bert_output)  # (b, 1), (b, n_filters * len(filter_sizes))
            res_sym_prob.append(symptom_prob)  # (num_symptoms, b, 1)
            res_sym_hidden.append(symptom_hidden)  # (num_symptoms, b, n_filters * len(filter_sizes))

        symptom_scores = torch.stack(res_sym_prob).transpose(0, 1)  # (b, num_symp, 1)
        symptom_vectors = torch.stack(res_sym_hidden).transpose(0, 1)  # (b, num_symp, n_fil * len(fs))

        return symptom_scores, sym_labels, symptom_vectors


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