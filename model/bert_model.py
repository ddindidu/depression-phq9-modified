import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from dataset import DepressionDataset

class BertModelforBaseline(nn.Module):
    def __init__(self, args, tokenizer, bert_model):
        super(BertModelforBaseline, self).__init__()

        self.bert_model = bert_model

    def forward(self, inputs, labels=None, output_hidden_states=False):
        return self.bert_model(**inputs,
                               output_hidden_states=output_hidden_states,
                               return_dict=True)


def get_batch_bert_embedding(bert_model, inputs, trainable=False):
    if not trainable:
        with torch.no_grad():
            output = bert_model(**inputs)
    else:
        output = bert_model(**inputs)

    return output['last_hidden_state']


if __name__ == '__main__':
    from train import get_args
    args = get_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    model = BertModelforBaseline(
        args,
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
        ),
        bert_model = AutoModel.from_pretrained(
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
        shuffle=True,
        pin_memory=True,
    )

    device = torch.device("cuda")
    model.cuda()

    for step, data in enumerate(train_dl):
        inputs = {
            "input_ids": data['input_ids'].to(device),
            "attention_mask": data['attention_mask'].to(device),
            "token_type_ids": data['token_type_ids'].to(device),
        }

        hidden = get_batch_bert_embedding(model, inputs, trainable=True)

        break

    import IPython; IPython.embed(); exit(1)