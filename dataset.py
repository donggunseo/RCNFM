import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import itertools
import json
import os
from tqdm import tqdm
import pickle

class RE_dataset(Dataset):
    def __init__(self, args, do_eval=False, do_test=False):
        if not do_eval:
            file_path = os.path.join(args.data_dir, 'train.json')
        elif not do_test:
            file_path = os.path.join(args.data_dir, 'dev.json')
        else:
            file_path = os.path.join(args.data_dir, 'test.json')
        assert os.path.isfile(file_path)

        self.file_path = file_path
        self.evaluate = do_eval
        self.test = do_test
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.data = []
        self.label = []
        self.id = []
        self.label_ori = []
        self.answer = []
        with open('../data/relation/l2i.json', 'r') as f:
            self.l2i = json.load(f)
        with open('../data/relation/entity_pair_text.pkl', 'rb') as f2:
            self.entity_pair_text = pickle.load(f2)
        self.initialize()
    def convert_bracket(self, token):
        bracket = {
            "-LRB-": "(",
            "-RRB-": ")",
            "-LSB-": "[",
            "-RSB-": "]",
            "-LCB-": "{",
            "-RCB-": "}",
        }
        if token in bracket:
            token = bracket[token]
        return token

    def initialize(self):
        with open(self.file_path, 'r') as file:
            data = json.load(file)
        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            id = d['id']
            tokens =  [self.convert_bracket(token) for token in d['token']]
            sentence = []
            subj_type_ori = d['subj_type']
            obj_type_ori = d['obj_type']
            subj_type = self.tokenizer.tokenize(d['subj_type'].replace("_", " ").lower())
            obj_type = self.tokenizer.tokenize(d['obj_type'].replace("_", " ").lower())
            for i, token in enumerate(tokens):
                t = self.tokenizer.tokenize(token)
                if i == ss:
                    t = ['@'] + ['*'] + subj_type + ['*'] + t
                if i == se:
                    t = t + ['@']
                if i == os:
                    t = ["#"] + ['^'] + obj_type + ['^'] + t
                if i == oe:
                    t = t + ["#"]
                sentence.extend(t)
            label = self.l2i[d['relation']]
            subj_entity = ['@'] + ['*'] + subj_type + ['*']
            obj_entity = ["#"] + ['^'] + obj_type + ['^']
            for j, token in enumerate(tokens[ss:se+1]):
                t = self.tokenizer.tokenize(token)
                subj_entity.extend(t)
            subj_entity.append('@')
            for j, token in enumerate(tokens[os:oe+1]):
                t = self.tokenizer.tokenize(token)
                obj_entity.extend(t)
            obj_entity.append('#')
            label_text_dict = self.entity_pair_text[(subj_type_ori,obj_type_ori)]
            label_list = [0 if self.l2i[k]!=label else 1 for k in label_text_dict.keys()]
            label_sentence_list = []
            for k,v in label_text_dict.items():
                label_text = v
                subj_idx = label_text.index('subj')
                obj_idx = label_text.index('obj')
                sentence2=[]
                for i, token in enumerate(label_text):
                    if i==subj_idx:
                        sentence2.extend(subj_entity)
                    elif i==obj_idx:
                        sentence2.extend(obj_entity)
                    else:
                        t = self.tokenizer.tokenize(token)
                        sentence2.extend(t)
                label_sentence_list.append(sentence2)
            s = [(sentence, k) for k in label_sentence_list]
            self.data.extend(s)
            self.label.extend(label_list)
            self.id.extend([id for _ in range(len(label_sentence_list))])
            self.label_ori.extend([self.l2i[k] for k in label_text_dict.keys()])
            self.answer.append(label)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        s1 = self.tokenizer.convert_tokens_to_ids(self.data[idx][0])
        s2 = self.tokenizer.convert_tokens_to_ids(self.data[idx][1])
        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        s = cls + s1 + sep + s2
        return {'input_ids': s, 'labels' : self.label[idx], 'id' : self.id[idx], 'label_ori' : self.label_ori[idx]}


