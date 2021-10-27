import os
import numpy as np
import torch
import torch.utils.data as data
import json
import re
import itertools
from collections import Counter
import pickle
import h5py
from parameters import PREPROCESSED_FILE_PATH, VOCABULARY_PATH, DATASET_PATH, questions_path, answers_path

ANSWER_VOCAB_LIMIT = 1000


comma_regex = re.compile(r'(\d)(,)(\d)')
punct_char_regex = re.escape(r';/[]"{}()=+\_-><@`,?!')
p_regex = re.compile(r'([{}])'.format(re.escape(punct_char_regex)))
space_regex = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(punct_char_regex))
specials_regex = re.compile('[^a-z0-9 ]*')
period_regex = re.compile(r'(?!<=\d)(\.)(?!\d)')


def prepare_questions(questions_json):
    """ Tokenize and normalize questions from a given question json in the usual VQA format. """
    for q in questions_json['questions']:
        question = q['question']
        question = question.lower()[:-1]
        question = process_punctuation(question)
        yield question.split(' ')


def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's
        if p_regex.search(s) is None:
            return s
        s = space_regex.sub('', s)
        if re.search(comma_regex, s) is not None:
            s = s.replace(',', '')
        s = p_regex.sub(' ', s)
        s = period_regex.sub('', s)
        return s.strip()


def prepare_answers(answers_json):
    for ans_dict in answers_json['annotations']:
        yield ans_dict['multiple_choice_answer']


class VqaDataset(data.Dataset):
    def __init__(self, vocabs, train=True):
        self.train = train
        self.question_vocab = vocabs['question']
        self.answer_vocab = vocabs['answer']
        self.q_length = vocabs['max_question_length']
        self.question_vocab_size = len(self.question_vocab) + 1 # for unknown words
        self.answer_vocab_size = len(self.answer_vocab) + 1
        self.questions = []
        self.top_answers = []
        self.answers = []
        self.iids = []
        self.answer_counts = []

        with h5py.File(PREPROCESSED_FILE_PATH, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        self.coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}

        with open(questions_path(train), 'r') as fd:
            questions_json = json.load(fd)
        for inner_index, q in enumerate(prepare_questions(questions_json)):
            vec = torch.zeros(self.q_length).long()
            self.iids.append(questions_json['questions'][inner_index]['image_id'])
            for i, token in enumerate(q):
                index = self.question_vocab.get(token, 0)
                vec[i] = index
            self.questions.append(vec)
        del questions_json

        with open(answers_path(train), 'r') as fd:
            answers_json = json.load(fd)
        to_remove = []
        for i, a_dict in enumerate(answers_json['annotations']):
            top_answer = a_dict['multiple_choice_answer']
            index = self.answer_vocab.get(top_answer)
            if index is not None:
                self.top_answers.append(index)
                self.answers.append([self.answer_vocab[ans['answer']] for ans in a_dict['answers'] if ans['answer'] in self.answer_vocab])
            else:
                if train:
                    to_remove.insert(0, i)
                else:
                    self.top_answers.append(self.answer_vocab_size - 1)
                    self.answers.append([])
        for i in to_remove:
            self.questions.pop(i)
            self.iids.pop(i)

        del answers_json

    def __getitem__(self, item):
        iid = self.iids[item]
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(PREPROCESSED_FILE_PATH, 'r')
        index = self.coco_id_to_index[iid]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        img = torch.from_numpy(img)
        if self.train:
            ans = np.random.choice(self.answers[item])
        else:
            ans = self.top_answers[item]
        result = {'question': self.questions[item], 'answer': ans, 'image': img, 'item_no': item}
        return result

    def __len__(self):
        return len(self.iids)


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if '' in counter:
        counter.pop('')
    if top_k:
        print('original count:', len(counter), 'cutting to', top_k)
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def get_vocab():
    if not os.path.exists(VOCABULARY_PATH):
        print('vocabularies not found, creating...')
        with open(questions_path(), 'r') as fd:
            questions = json.load(fd)
        with open(answers_path(), 'r') as fd:
            answers = json.load(fd)

        prepared_q = prepare_questions(questions)
        prepared_a = prepare_answers(answers)

        question_vocab = extract_vocab(prepared_q, start=1)
        counts = Counter(prepared_a)
        top_answers = [x[0] for x in sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)][:ANSWER_VOCAB_LIMIT]
        answer_vocab = {t: i for i,t in enumerate(sorted(top_answers))}
        print('Cutting answer vocabulary to: ', len(answer_vocab))
        vocabs = {
            'question': question_vocab,
            'answer': answer_vocab,
            'max_question_length': max([len(x) for x in prepare_questions(questions)]) + 1
        }
        with open(VOCABULARY_PATH, 'w') as fd:
            json.dump(vocabs, fd)
    else:
        with open(VOCABULARY_PATH, 'r') as fd:
            vocabs = json.load(fd)
    return vocabs


def get_datasets(vocabs):
    if not os.path.exists(DATASET_PATH):
        print('Creating datasets')
        train_data = VqaDataset(vocabs, True)
        val_data = VqaDataset(vocabs, False)
        with open(DATASET_PATH, 'wb') as f:
            pickle.dump([train_data, val_data], f)
    else:
        print("Loading datasets")
        try:
            with open(DATASET_PATH, 'rb') as f:
                train_data, val_data = pickle.load(f)
        except AttributeError:
            print('Creating datasets')
            train_data = VqaDataset(vocabs, True)
            val_data = VqaDataset(vocabs, False)
            with open(DATASET_PATH, 'wb') as f:
                pickle.dump([train_data, val_data], f)
    return train_data, val_data

