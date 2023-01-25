import os, argparse, json


def read_json(args):
    data_path = args.data_path.format(args.task_name)
    file_name = ''
    if args.task_name == 'bipolar':
        file_name = 'questions.json'
    else:
        # task_name == depression, bpd, anxiety
        file_name = 'questions2.json'

    data_path = os.path.join(data_path, file_name)
    assert os.path.exists(data_path), "DATA_PATH DOESN'T EXIST!"

    with open(data_path, 'r') as f:
        datas = json.load(f)

        texts = []
        labels = []
        for idx, label in enumerate(datas):
            for txt in datas[label]:
                texts.append(txt)
                labels.append(label)
    return texts, labels


def get_label_number(task_name, label_name):
    if task_name == 'bipolar':
        if label_name == 'major_depressive_episode':
            return '0'
        elif label_name == 'manic_episode':
            return '1'
        else:
            # label_name == 'comorbidity'
            return '2'
    else:   # depression, bpd, anxiety
        return label_name   # "n"


def process_dataset(args):
    texts, labels = read_json(args)

    json_dict = dict()
    for idx, data in enumerate(zip(texts, labels)):
        txt = data[0]
        lab = data[1]

        num_label = get_label_number(args.task_name, lab)

        json_dict[idx] = [txt, num_label]

    data_path = args.data_path.format(args.task_name)
    save_path = os.path.join(data_path, 'train.json')

    with open(save_path, 'w') as outf:
        json.dump(json_dict, outf)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./{}')

    parser.add_argument('--task_name', type=str, default='depression')

    return parser.parse_args()

if __name__ == "__main__":
    from process_questionnaire import get_args
    args = get_args()

    process_dataset(args)
    #data = read_json(args)

    #import IPython;
    #IPython.embed();
    #exit(1)