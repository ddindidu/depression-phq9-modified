import time, datetime, random, os
import numpy as np

from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, auc, roc_curve, confusion_matrix)
from transformers import get_linear_schedule_with_warmup
import torch


class InputExample(object):
    """
    A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b, label, features=[]):
        self.guid = guid
        self.label = label
        self.text_a = text_a
        self.text_b = text_b
        self.features = features

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def generate_examples(mode, texta=[], textb=None, labels=[]):
    examples = []

    for idx in range(len(texta)):
        guid = "%s-%s" % (mode, idx)
        text_a = texta[idx]
        if textb is not None:
            text_b = textb[idx]
        else:
            text_b = None
        label = labels[idx]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )

    return examples



def f1_pre_rec_scalar(labels, preds, main_label=2):
    fpr, tpr, thresholds = roc_curve(labels, preds, pos_label=main_label)
                            #roc_curve(np.sort(labels), np.sort(preds), pos_label=main_label)
    return {
        "acc": simple_accuracy(labels, preds),
        #"precision_micro": precision_score(labels, preds, average="micro"),
        #"recall_micro": recall_score(labels, preds, average="micro"),
        #"f1_micro": f1_score(labels, preds, average="micro"),
        "precision": precision_score(labels, preds, average=None)[main_label],
        "recall": recall_score(labels, preds, average=None)[main_label],
        "f1": f1_score(labels, preds, average=None)[main_label],

        "precision_weighted": precision_score(labels, preds, average="weighted"),
        "recall_weighted": recall_score(labels, preds, average="weighted"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),

        "f1_macro": f1_score(labels, preds, average="macro"),
        "AUC": auc(fpr, tpr),
    }, confusion_matrix(labels, preds)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec_scalar(labels, preds)
    

def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def print_result(test_result):
	for name, value in test_result.items():
		print('  Average {}:\t{}'.format(name, round(value*100, 4)))


def save_cp(args, model_name, batch_size, epochs, model, optimizer, scheduler, tokenizer):
    if model_name == 'question_model':
        m_name = 'symptoms'
    elif model_name == 'disease_model':
        m_name = 'disease'
    save_dir_path = os.path.join(args.output_dir,   # './checkpoints'
                                 '{}/{}/{}/checkpoint_batch_{}_ep_{}/'.format(
                                     m_name,
                                     args.task_name,
                                     args.model_name_or_path,
                                     batch_size,
                                     epochs+1)
                                 )

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    print('*** Save checkpoints at {}'.format(save_dir_path))
    torch.save(model, save_dir_path + 'model.bin')
    torch.save(optimizer, save_dir_path + 'optimizer.pt')
    torch.save(scheduler.state_dict(), save_dir_path + 'scheduler.pt')
    torch.save(tokenizer, save_dir_path + 'tokenizer.json')


def save_cp_epochs(args, batch_size, epochs, model, optimizer, scheduler, tokenizer):
    if args.train_portion == 100:
        # path = "./checkpoints/data_type/task_type/...model_name.../checkpoint_{batch_size}_{epochs}/
        save_dir_path = os.path.join(args.output_dir,
                                    '{}/{}/{}/checkpoint_{}_{}/'.format(args.data_type,
                                                                         args.task_type,
                                                                        args.model_name_or_path,
                                                                        batch_size, epochs))

    else:  #if train_portion != 100
        # path = # path = "./checkpoints/data_type/task_type/...model_name.../checkpoint_{train_portion}_{batch_size}_{epochs}/
        save_dir_path = os.path.join(args.output_dir,
                                     '{}/{}/{}/checkpoint_{}_{}_{}/'.format(args.data_type,
                                                                            args.task_type,
                                                                            args.model_name_or_path,
                                                                            args.train_portion, batch_size, epochs))

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    print('*** Saving checkpoints at {}'.format(save_dir_path))
    torch.save(model, save_dir_path+'model.bin')
    torch.save(optimizer, save_dir_path+'optimizer.pt')
    torch.save(scheduler.state_dict(), save_dir_path+'scheduler.pt')
    torch.save(tokenizer, save_dir_path+'tokenizer.json')


def save_cp_steps(args, batch_size, steps, model, optimizer, scheduler, tokenizer):
    if args.train_portion == 100:
        # path = "./checkpoints/data_type/task_type/...model_name.../checkpoint_{batch_size}_{steps}steps/
        save_dir_path = os.path.join(args.output_dir,
                                    '{}/{}/{}/checkpoint_{}_{}steps/'.format(args.data_type,
                                                                         args.task_type,
                                                                        args.model_name_or_path,
                                                                        batch_size, steps))

    else:  #if train_portion != 100
        # path = # path = "./checkpoints/data_type/task_type/...model_name.../checkpoint_{train_portion}_{batch_size}_{epochs}/
        save_dir_path = os.path.join(args.output_dir,
                                     '{}/{}/{}/checkpoint_{}_{}_{}steps/'.format(args.data_type,
                                                                            args.task_type,
                                                                            args.model_name_or_path,
                                                                            args.train_portion, batch_size, steps))

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    print('*** Saving checkpoints at {}'.format(save_dir_path))
    torch.save(model, save_dir_path+'model.bin')
    torch.save(optimizer, save_dir_path+'optimizer.pt')
    torch.save(scheduler.state_dict(), save_dir_path+'scheduler.pt')
    torch.save(tokenizer, save_dir_path+'tokenizer.json')


def load_tokenizer(path):
    return torch.load(path+'tokenizer.json')


def load_model(path):
    return torch.load(path+'model.bin')


def load_optimizer(path):
    return torch.load(path+'optimizer.pt')


def load_scheduler(path, optimizer, warmup_steps, num_training_steps):
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    scheduler.load_state_dict(torch.load(path+'scheduler.pt'))
    
    return scheduler


def get_symptom_num(task_name):
    if task_name == 'depression':
        return 9
    elif task_name == 'bpd':
        return 9
    elif task_name == 'bipolar':
        return 3
    elif task_name == 'anxiety':
        return 7

