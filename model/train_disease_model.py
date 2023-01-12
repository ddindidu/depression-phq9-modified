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
from utils import save_cp, format_time, load_model, compute_metrics, print_result, get_symptom_num
from bert_model import BertModelforBaseline, get_batch_bert_embedding
from questionnaire.questionnaire_model import QuestionnaireModel
from disease.disease_model import DiseaseModel, DiseaseAfterBertModel, DiseaseModelfor2Inputs


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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
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
    parser.add_argument("--do_train", action="store_true")  # only True when entered in an argument line
    #parser.add_argument("--do_eval", action="store_true", default=True)
    parser.add_argument("--do_test", action="store_true", default=True)

    return parser.parse_args()


def train(args):
    #print(args)
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
    train_dataset = DepressionDataset(
        args=args,
        mode='train',
        tokenizer=tokenizer,
    )

    test_dataset = DepressionDataset(
        args=args,
        mode='test',
        tokenizer=tokenizer,
    )

    # Load Data
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    dataloaders = {
        'train': train_dl,
        'test': test_dl
    }

    # Prepare models
    num_training_steps = args.epochs * (train_dataset.num_data / args.batch_size)
    # BERT Encoder
    bert_model = BertModelforBaseline(
        args=args,
        tokenizer=tokenizer,
        bert_model=AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
        ),
    )
    # questionnaire model
    '''
    question_model_path = os.path.join(args.output_dir,  # './checkpoints'
                                 '{}/{}/{}/checkpoint_batch_{}_ep_{}/'.format(
                                     'symptoms',
                                     args.task_name,
                                     args.model_name_or_path,
                                     args.batch_size,
                                     '49')
                                 )
    question_model = load_model(question_model_path)
    '''
    # disease model (depression model in code)
    disease_model = DiseaseAfterBertModel()

    bert_model.cuda()
    #question_model.cuda()
    disease_model.cuda()

    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    print("BERT MODEL PARAMS: {}".format(count_parameter(bert_model)))
    #print("QUESTION MODEL PARAMS: {}".format(count_parameter(question_model)))
    print("DISEASE MODEL PARAMS: {}".format(count_parameter(disease_model)))

    optimizer = torch.optim.AdamW(
        disease_model.parameters(),
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

    start_time = time.time()
    # Initialize a counter for the number of iterations
    iteration_count = 0
    # Initialize a counter for the number of steps
    step_count = 0

    for epoch_i in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))


        phases = ['train', 'test'] if args.do_train else ['test']

        for phase in phases:
            if (phase == 'test') and (epoch_i != args.epochs-1):
                continue
            t0 = time.time()

            # train starts
            print('{}ing...'.format(phase))
            total_loss = 0.0
            loss_for_logging = 0.0

            all_preds = []
            all_labels = []
            for step, data in enumerate(tqdm(dataloaders[phase], desc=phase, mininterval=0.01, leave=True), 0):
                inputs = {
                    "input_ids": data['input_ids'].to(device),
                    "attention_mask": data['attention_mask'].to(device),
                    # "token_type_ids":data['token_type_ids'].to(device),
                }
                labels = data['labels'].to(device)

                optimizer.zero_grad()

                # foward
                with torch.no_grad():
                    bert_output = get_batch_bert_embedding(bert_model, inputs, trainable=False)
                    #symptom_scores, symptom_labels, symptom_hidden = question_model.forward(bert_output,
                    #                                                                        labels)  # (b, num_symptom, 1), (b, num_symptom, 1), (b, 5)
                with torch.set_grad_enabled(phase == 'train'):
                    #disease_output, disease_hidden = disease_model(symptom_hidden)  # (b, 1), (b, hidden_dim)
                    disease_output, disease_hidden = disease_model(bert_output) # (b, 1), (b, hidden_dim)
                preds = [1 if prob.item() >= 0.5 else 0 for prob in disease_output]

                loss = loss_fn(disease_output.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
                total_loss += loss.item()
                loss_for_logging += loss.item()

                if phase == 'train':
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(disease_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                # logging
                if step % args.logging_steps == 0 and not step == 0:
                    writer.add_scalar('{}/loss'.format(phase), (loss_for_logging / args.logging_steps), total_train_step)
                    loss_for_logging = 0

                #elapsed = format_time(time.time() - t0)
                #print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.    Loss: {}'.format(step, len(dataloaders[phase]), elapsed, loss))

                # when step ends
                total_train_step += 1
                #question_model.zero_grad()
                all_preds += preds
                all_labels += labels.tolist()

                # Increment the iteration counter
                iteration_count += 1
                # Increment the step counter
                step_count += 1

                # import IPython; IPython.embed(); exit(1)

            # epoch ends
            # print results
            print("total {} loss: {}".format(phase, total_loss / len(dataloaders[phase])))

            if (epoch_i == args.epochs-1) and (phase == 'test'):
                print("Test Result for\nTASK {} / MODEL {} / SEED {} / EP {} / FIVE FOLD {}".format(args.task_name,
                                                                                                   args.model_name_or_path,
                                                                                                   args.seed,
                                                                                                   args.epochs,
                                                                                                   args.five_fold_num))
                train_result, conf_matrix = compute_metrics(labels=all_labels, preds=all_preds)
                print_result(train_result)
                #print("Confusion Matrix:\n", conf_matrix)
            print("  {} epoch took: {:}".format(phase, format_time(time.time() - t0)))
            # save checkpoint
            if (epoch_i == args.epochs-1) and (phase == 'train'):
                save_cp(args=args,
                        model_name='disease_model',
                        seed=args.seed,
                        epochs=epoch_i,
                        fold=args.five_fold_num,
                        model=disease_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        tokenizer=tokenizer
                        )

    print("")
    print("Training complete")

    end_time = time.time()
    # Calculate the time per iteration
    time_per_iteration = (end_time - start_time) / iteration_count

    # Calculate the iterations per second
    iterations_per_second = 1 / time_per_iteration
    # Print the iterations per second
    print(f'Iterations per second: {iterations_per_second:.4f}')
    # Calculate the time per step
    time_per_step = (end_time - start_time) / step_count

    # Print the time per step
    print(f'Time per step: {time_per_step:.4f} seconds')


def test_only(args):
    #print(args)
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

    test_dataset = DepressionDataset(
        args=args,
        #mode = 'test',
        #mode='rsdd_test',
        mode='eRisk2018_test',
        tokenizer=tokenizer,
    )

    # Load Data
    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    dataloaders = {
        'test': test_dl
    }

    # Prepare models
    #num_training_steps = args.epochs * (train_dataset.num_data / args.batch_size)
    # BERT Encoder
    bert_model = BertModelforBaseline(
        args=args,
        tokenizer=tokenizer,
        bert_model=AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
        ),
    )
    # questionnaire model
    '''
    question_model_path = os.path.join(args.output_dir,  # './checkpoints'
                                 '{}/{}/{}/checkpoint_batch_{}_ep_{}/'.format(
                                     'symptoms',
                                     args.task_name,
                                     args.model_name_or_path,
                                     args.batch_size,
                                     '49')
                                 )
    question_model = load_model(question_model_path)
    '''
    # disease model (depression model in original paper)
    disease_model_path = os.path.join(args.output_dir,
                                      '{}/{}/{}/checkpoint_seed_{}_ep_{}_fivefold_{}/'.format(
                                         'disease',
                                         args.task_name,
                                         args.model_name_or_path,
                                         args.seed,
                                         args.epochs,
                                         args.five_fold_num)
                                     )
    disease_model = load_model(disease_model_path)

    bert_model.cuda()
    #question_model.cuda()
    disease_model.cuda()

    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    print("BERT MODEL PARAMS: {}".format(count_parameter(bert_model)))
    #print("QUESTION MODEL PARAMS: {}".format(count_parameter(question_model)))
    print("DISEASE MODEL PARAMS: {}".format(count_parameter(disease_model)))

    loss_fn = nn.BCELoss()

    # Training starts
    total_train_step = 0
    total_valid_step = 0

    for epoch_i in range(1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))

        phases = ['train', 'test'] if args.do_train else ['test']

        for phase in phases:
            t0 = time.time()
            # train starts
            print('{}ing...'.format(phase))
            total_loss = 0.0
            loss_for_logging = 0.0

            all_preds = []
            all_labels = []
            for step, data in enumerate(tqdm(dataloaders[phase], desc=phase, mininterval=0.01, leave=True), 0):
                inputs = {
                    "input_ids": data['input_ids'].to(device),
                    "attention_mask": data['attention_mask'].to(device),
                    # "token_type_ids":data['token_type_ids'].to(device),
                }
                labels = data['labels'].to(device)

                #optimizer.zero_grad()

                # foward
                with torch.no_grad():
                    bert_output = get_batch_bert_embedding(bert_model, inputs, trainable=False)
                    #symptom_scores, symptom_labels, symptom_hidden = question_model.forward(bert_output,
                    #                                                                        labels)  # (b, num_symptom, 1), (b, num_symptom, 1), (b, 5)
                with torch.set_grad_enabled(phase == 'train'):
                    #disease_output, disease_hidden = disease_model(symptom_hidden)  # (b, 1), (b, hidden_dim)
                    disease_output, disease_hidden = disease_model(bert_output) # (b, 1), (b, hidden_dim)
                preds = [1 if prob.item() >= 0.5 else 0 for prob in disease_output]

                loss = loss_fn(disease_output.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
                total_loss += loss.item()
                loss_for_logging += loss.item()

                # logging
                #if step % args.logging_steps == 0 and not step == 0:
                #    writer.add_scalar('{}/loss'.format(phase), (loss_for_logging / args.logging_steps), total_train_step)
                #    loss_for_logging = 0

                # when step ends
                total_train_step += 1
                #question_model.zero_grad()
                all_preds += preds
                all_labels += labels.tolist()
                # import IPython; IPython.embed(); exit(1)

            # epoch ends
            # print results
            print("total {} loss: {}".format(phase, total_loss / len(dataloaders[phase])))

            print("Test Result\nTASK {} / MODEL {} / SEED {} / FIVE FOLD {}".format(args.task_name,
                                                                                            args.model_name_or_path,
                                                                                            args.seed,
                                                                                            args.five_fold_num))
            train_result, conf_matrix = compute_metrics(labels=all_labels, preds=all_preds)
            print_result(train_result)
            print("")
            #print("Confusion Matrix:\n", conf_matrix)
            #print("  {} epoch took: {:}".format(phase, format_time(time.time() - t0)))

    #print("")
    #print("Test complete")


def train_for_measuring_time(args):
    #print(args)
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
    train_dataset = DepressionDataset(
        args=args,
        mode='train',
        tokenizer=tokenizer,
    )

    test_dataset = DepressionDataset(
        args=args,
        mode='test',
        tokenizer=tokenizer,
    )

    # Load Data
    train_dl = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    dataloaders = {
        'train': train_dl,
        'test': test_dl
    }

    # Prepare models
    num_training_steps = args.epochs * (train_dataset.num_data / args.batch_size)
    # BERT Encoder
    bert_model = BertModelforBaseline(
        args=args,
        tokenizer=tokenizer,
        bert_model=AutoModel.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            num_labels=args.num_labels,
        ),
    )
    # question model
    question_model_path = os.path.join(args.output_dir,  # './checkpoints'
                                       '{}/{}/{}/checkpoint_batch_{}_ep_{}/'.format(
                                           'symptoms',
                                           args.task_name,
                                           args.model_name_or_path,
                                           args.batch_size,
                                           '10')
                                       )
    question_model = load_model(question_model_path)
    # disease model (depression model in original paper)
    disease_model = DiseaseModelfor2Inputs(num_symptom=args.num_labels)

    bert_model.cuda()
    question_model.cuda()
    disease_model.cuda()

    def count_parameter(model):
        return sum(p.numel() for p in model.parameters())
    m_name = 'BERT' if args.model_name_or_path=='bert-base-cased' else 'ROBERTA'
    print("{} MODEL PARAMS: {}".format(m_name, count_parameter(bert_model)))
    print("QUESTION MODEL PARAMS: {}".format(count_parameter(question_model)))
    print("DISEASE MODEL PARAMS: {}".format(count_parameter(disease_model)))

    optimizer = torch.optim.AdamW(
        disease_model.parameters(),
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

    start_time = time.time()
    # Initialize a counter for the number of iterations
    iteration_count = 0
    # Initialize a counter for the number of steps
    step_count = 0
    memory_usage = []

    for epoch_i in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))


        phases = ['train'] if args.do_train else ['test']

        for phase in phases:
            if (phase == 'test') and (epoch_i != args.epochs-1):
                continue
            t0 = time.time()

            # train starts
            print('{}ing...'.format(phase))
            total_loss = 0.0
            loss_for_logging = 0.0

            all_preds = []
            all_labels = []
            for step, data in enumerate(tqdm(dataloaders[phase], desc=phase, mininterval=0.01, leave=True), 0):
                inputs = {
                    "input_ids": data['input_ids'].to(device),
                    "attention_mask": data['attention_mask'].to(device),
                    # "token_type_ids":data['token_type_ids'].to(device),
                }
                labels = data['labels'].to(device)

                # foward
                with torch.no_grad():
                    bert_output = get_batch_bert_embedding(bert_model, inputs, trainable=False)
                    symptom_scores, symptom_labels, symptom_hidden = question_model.forward(bert_output,
                                                                                            labels)  # (b, num_symptom, 1), (b, num_symptom, 1), (b, 5)
                with torch.set_grad_enabled(phase == 'train'):
                    disease_output, disease_hidden = disease_model(bert_output, symptom_hidden)  # (b, 1), (b, hidden_dim)
                    # disease_output, disease_hidden = disease_model(bert_output) # (b, 1), (b, hidden_dim)
                preds = [1 if prob.item() >= 0.5 else 0 for prob in disease_output]

                loss = loss_fn(disease_output.to(torch.float32), labels.unsqueeze(1).to(torch.float32))
                total_loss += loss.item()
                loss_for_logging += loss.item()

                if phase == 'train':
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(disease_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    optimizer.zero_grad()

                # when step ends
                total_train_step += 1
                all_preds += preds
                all_labels += labels.tolist()

                # Increment the iteration counter
                iteration_count += 1
                # Increment the step counter
                step_count += 1
                # Record memory usage
                memory_usage.append(torch.cuda.memory_allocated(device) / (1024 ** 2))

                # import IPython; IPython.embed(); exit(1)
            train_result, conf_matrix = compute_metrics(labels=all_labels, preds=all_preds)
            print_result(train_result)
            # epoch ends
            # print results

    print("")
    print("Training complete")

    end_time = time.time()
    # Calculate the time per iteration
    time_per_iteration = (end_time - start_time) / iteration_count

    # Calculate the iterations per second
    iterations_per_second = 1 / time_per_iteration
    # Print the iterations per second
    print(f'Iterations per second: {iterations_per_second:.4f}')
    # Calculate the time per step
    time_per_step = (end_time - start_time) / step_count

    # Print the time per step
    print(f'Time per step: {time_per_step:.4f} seconds')

    # Calculate and print average memory usage per iteration
    average_memory_usage = sum(memory_usage) / len(memory_usage)
    print(f"Average memory usage per iteration: {average_memory_usage:.2f} MB")




if __name__ == '__main__':
    from train_disease_model import get_args

    args = get_args()
    args.num_labels = get_symptom_num(args.task_name)

    if args.do_train:
        #train(args)
        train_for_measuring_time(args)
    else:
        test_only(args)