import numpy as np
import os, sys, argparse, time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    BertModel,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    set_seed,
    get_linear_schedule_with_warmup,
)

from tqdm import tqdm

from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)

from dataset import DepressionDataset, SymptomDataset
from utils import save_cp, format_time, compute_metrics, print_result, get_symptom_num, load_model
from bert_model import BertModelforBaseline, get_batch_bert_embedding
from questionnaire.questionnaire_model import QuestionnaireModel


def get_args():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_path", type=str, default="./dataset/{}/{}/{}.json")
    parser.add_argument("--five_fold_num", type=int, default=0)

    # dataset related
    parser.add_argument('--num_labels', type=int, default=2)

    # model related
    parser.add_argument("--project_name", type=str, default="proposed")
    parser.add_argument("--task_name", type=str, default="depression")
    parser.add_argument("--kernel_size", nargs='+', type=int, default=[2, 3, 4, 5, 6])

    parser.add_argument("--model_name", type=str, default="bert")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    # parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    # parser.add_argument("--model_name_or_path", type=str, default="xlnet-base-cased")

    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument("--warmup_steps", type=int, default=1100)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=1000)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--betas", type=float, default=(0.9, 0.98), nargs='+')

    # trainer related
    parser.add_argument("--load_from_checkpoint", type=str, default="")
    parser.add_argument('--debug', action='store_true')

    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument('--fast_dev_run', action='store_true', default=True)
    parser.add_argument("--do_train", action="store_true", default=True)
    parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--do_test", action="store_true", default=True)

    return parser.parse_args()


def main(args):
    print(args)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    writer = SummaryWriter(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('  *** Device: ', device)
    print('  *** Current cuda device:', args.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    # Prepare data
    train_dataset = SymptomDataset(
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

    # Prepare models
    num_training_steps = args.epochs * (train_dataset.num_data / args.batch_size)

    bert_model = BertModelforBaseline(
        args=args,
        tokenizer=tokenizer,
        bert_model=AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
        ),
    )

    question_model = QuestionnaireModel(
        num_symptoms=get_symptom_num(args.task_name),
        filter_sizes=args.kernel_size,
    )

    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    print("BERT MODEL PARAMS: {}".format(count_parameter(bert_model)))#, "ERROR in PARAMS COUNTING"
    print("QUESTION MODEL PARAMS: {}".format(count_parameter(question_model)))

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
        num_training_steps=num_training_steps,
    )

    loss_fn = nn.BCELoss()

    # Training starts
    total_train_step = 0
    total_valid_step = 0

    for epoch_i in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))

        t0 = time.time()

        # train starts
        print('Training...')
        total_loss = 0.0
        loss_for_logging = 0.0

        for step, data in enumerate(tqdm(train_dl, desc='train', mininterval=0.01, leave=True), 0):
            inputs = {
                "input_ids": data['input_ids'].to(device),
                "attention_mask": data['attention_mask'].to(device),
                # "token_type_ids":data['token_type_ids'].to(device),
            }
            labels = data['labels'].to(device)

            optimizer.zero_grad()

            # foward
            bert_output = get_batch_bert_embedding(bert_model, inputs, trainable=True)
            symptom_scores, symptom_labels, symptom_hidden = question_model.forward(bert_output,
                                                                                    labels)  # (b, num_symptom, 1), (b, num_symptom, 1), (b, 5)

            loss = loss_fn(symptom_scores.to(torch.float32), symptom_labels.to(torch.float32).to(device))
            total_loss += loss.item()
            loss_for_logging += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(bert_model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(question_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # logging
            if step % args.logging_steps == 0 and not step == 0:
                writer.add_scalar('Train/loss', (loss_for_logging / args.logging_steps), total_train_step)
                loss_for_logging = 0

            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}'.format(step, len(train_dl), elapsed, loss))

            # save checkpoint
            """
            if total_train_step % args.saving_steps == 0 and not total_train_step==0 and total_train_step==2000:
                save_cp(args, args.batch_size, epoch_i, 
                    total_train_step,
                    model, 
                    optimizer, 
                    scheduler, 
                    tokenizer
                )
            """

            # when step ends
            total_train_step += 1
            # question_model.zero_grad()
            # import IPython; IPython.embed(); exit(1)

        # epoch ends ... print results
        print("total train loss: {}".format(total_loss / len(train_dl)))
        # train_result = compute_metrics(labels=all_labels, preds=all_preds)
        # print_result(train_result)
        print("  Train epoch took: {:}".format(format_time(time.time() - t0)))

        # save checkpoint
        if (epoch_i+1) % 5 == 0:
            save_cp(args=args,
                    model_name='question_model',
                    epochs=epoch_i,
                    fold=0,
                    model=question_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    tokenizer=tokenizer,
                    batch_size=args.batch_size,)

    print("")
    print("Training complete")


def test_only(args):
    print(args)
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    writer = SummaryWriter(args.log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('  *** Device: ', device)
    print('  *** Current cuda device:', args.gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    # Prepare data
    test_dataset = SymptomDataset(
        args=args,
        #mode='test',
        mode='analysis',
        tokenizer=tokenizer,
    )

    # Load Data
    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    # Prepare models
    num_training_steps = args.epochs * (test_dataset.num_data / args.batch_size)

    bert_model = BertModelforBaseline(
        args=args,
        tokenizer=tokenizer,
        bert_model=AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
        ),
    )

    question_model_path = os.path.join(args.output_dir,  # './checkpoints'
                                       '{}/{}/{}/checkpoint_batch_{}_ep_{}/'.format(
                                           'symptoms',
                                           args.task_name,
                                           args.model_name_or_path,
                                           args.batch_size,
                                            args.epochs)
                                       )
    question_model = load_model(question_model_path)

    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    print("BERT MODEL PARAMS: {}".format(count_parameter(bert_model)))#, "ERROR in PARAMS COUNTING"
    print("QUESTION MODEL PARAMS: {}".format(count_parameter(question_model)))

    bert_model.cuda()
    question_model.cuda()

    loss_fn = nn.BCELoss()

    # Training starts
    total_train_step = 0
    total_valid_step = 0

    print("")
    print('======== Epoch {:} / {:} ========'.format(1, args.epochs))

    t0 = time.time()

    # train starts
    print('Training...')
    total_loss = 0.0
    loss_for_logging = 0.0

    for step, data in enumerate(tqdm(test_dl, desc='test', mininterval=0.01, leave=True), 0):
        inputs = {
            "input_ids": data['input_ids'].to(device),
            "attention_mask": data['attention_mask'].to(device),
            # "token_type_ids":data['token_type_ids'].to(device),
        }
        labels = data['labels'].to(device)

        # foward
        bert_output = get_batch_bert_embedding(bert_model, inputs, trainable=True)
        symptom_scores, symptom_labels, symptom_hidden = question_model.forward(bert_output,
                                                                                labels)  # (b, num_symptom, 1), (b, num_symptom, 1), (b, 5)

        label_shape = symptom_labels.size()
        print("Symptom_Labels\n", symptom_scores.view(label_shape[:-1]))

        loss = loss_fn(symptom_scores.to(torch.float32), symptom_labels.to(torch.float32).to(device))
        total_loss += loss.item()
        loss_for_logging += loss.item()

        # logging
        if step % args.logging_steps == 0 and not step == 0:
            writer.add_scalar('Train/loss', (loss_for_logging / args.logging_steps), total_train_step)
            loss_for_logging = 0

        elapsed = format_time(time.time() - t0)
        print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}'.format(step, len(test_dl), elapsed, loss))


        # when step ends
        total_train_step += 1
        # question_model.zero_grad()
        # import IPython; IPython.embed(); exit(1)

    # epoch ends ... print results
    print("total teste loss: {}".format(total_loss / len(test_dl)))
    # train_result = compute_metrics(labels=all_labels, preds=all_preds)
    # print_result(train_result)
    print("  Test epoch took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Testing complete")


if __name__ == '__main__':
    from train_question_model import get_args

    args = get_args()
    args.num_labels = get_symptom_num(args.task_name)

    #main(args)
    test_only(args)