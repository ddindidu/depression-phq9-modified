import argparse, os
import numpy as np


def get_args_for_bash():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument("--model", type=str, default='bert')
    parser.add_argument('--gpu_id', type=str, default="0")
    parser.add_argument('--task_name', type=str, default='depression')
    parser.add_argument('--seed', nargs='+', type=int, default=[42, 53, 64, 75, 86, 97])
    parser.add_argument("--do_train", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    from run_disease_model import get_args_for_bash
    args = get_args_for_bash()

    model_name = ''
    if args.model == 'bert':
        model_name = 'bert-base-cased'
    elif args.model == 'roberta':
        model_name = 'roberta-base'

    random_seed = args.seed

    if args.do_train:
        command = "python train_disease_model.py --task_name '{}' --batch_size=32 --epochs=3 --gpu_id '{}' --do_train --lr 1e-3 --model_name_or_path '{}' --seed {} --five_fold_num {}"
    else:
        command = "python train_disease_model.py --task_name '{}' --batch_size=32 --epochs=3 --gpu_id '{}' --model_name_or_path '{}' --seed {} --five_fold_num {}"

    for seed in random_seed:
        for fold in range(5):
            new_command = command.format(args.task_name,
                                         args.gpu_id,
                                         model_name,
                                         seed,
                                         fold)
            # print(new_command)
            os.system(new_command)
