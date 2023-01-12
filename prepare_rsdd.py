import os, re, json


def select_10_sen(mode):
    with open(mode, 'r') as file:
        lines = file.readlines()

    new_data = dict()
    data_idx = 0
    for line in lines:
        line = line.strip()[1:-1]
        new_line = json.loads(line)  # {'posts': [#, string, ]
        assert len(new_line.keys()) == 2

        label = new_line['label']
        label_num = 0 if label == 'control' else 1

        posts = new_line['posts']
        # depression user
        if label_num == 1:
            posts = sorted(posts, key=len, reverse=True)

        # post indices
        end = 10 if len(posts) > 10 else len(posts)
        if label_num == 0:
            import random
            idx = random.sample(range(0, len(posts)), end)
        else:
            idx = range(0, end)
        selected_posts = [posts[i] for i in idx]

        # select less than 10 posts
        for post in selected_posts:
            txt = post[1]
            new_data[data_idx] = [txt, label_num]
            data_idx += 1

    print(len(new_data))

    with open('test_10.json', 'w') as new_file:
        json.dump(new_data, new_file)


def select_concat_sen(mode):
    with open(mode, 'r') as file:
        lines = file.readlines()

    new_data = dict()
    data_idx = 0
    for line in lines:
        line = line.strip()[1:-1]
        new_line = json.loads(line)  # {'posts': [#, string, ]
        assert len(new_line.keys()) == 2

        label = new_line['label']
        label_num = 0 if label == 'control' else 1

        full_sen = []
        for post in new_line['posts']:
            full_sen += post[1].split()
        n_tokens = len(full_sen)

        sen_len = 400
        posts = []
        for i in range(0, n_tokens, sen_len):
            start = i
            end = start + sen_len if start+sen_len < n_tokens else n_tokens

            post = ' '.join(full_sen[start:end])
            posts.append(post)

        # post indices
        num_post = 10 if len(posts) > 10 else len(posts)
        if label_num == 0:
            import random
            #idx = random.sample(range(0, len(posts)), num_post)
            idx = random.sample(range(0, len(posts)), 1)
        else:
            idx = range(0, num_post)
        selected_posts = [posts[i] for i in idx]

        # select less than 10 posts
        for post in selected_posts:
            txt = post
            new_data[data_idx] = [txt, label_num]
            data_idx += 1

    print(len(new_data))

    with open('test_concat_long_balanced.json', 'w') as new_file:
        json.dump(new_data, new_file)


if __name__ == '__main__':
    path = '/home/hysong/Research/dep_detection/rsdd/rsdd_posts'
    os.chdir(path)

    # mode = ['training', 'validation', 'testing']
    modes = ['testing']

    for mode in modes:
        select_concat_sen(mode)