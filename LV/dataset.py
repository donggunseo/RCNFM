from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import numpy as np
import os
from tqdm import tqdm
import random
from collections import defaultdict

class RE2NLI_dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.file_path = "../../data/tacred/train.json"
        self.data = []
        with open(self.file_path, 'r') as f:
            for i, line in tqdm(enumerate(json.load(f))):
                subj = " ".join(line["token"][line["subj_start"] : line["subj_end"] + 1]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
                obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
                inst_type=f"{line['subj_type']}:{line['obj_type']}"
                context=" ".join(line["token"]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
                label=line["relation"]
                self.data.append({'subj' : subj, 'obj' : obj, 'inst_type' : inst_type, 'context' : context, 'label' : label})
        
        with open("./LV_template.json", 'r') as f:
            LV_template = json.load(f)
        self.labels = LV_template['labels']
        self.templates = LV_template['templates']
        self.valid_conditions = LV_template['valid_conditions']
        self.negative_label_id = 0

        self.label2id = {label : i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)
        self.template2label = defaultdict(list)
        for label, templates in self.templates.items():
            for template in templates:
                self.template2label[template].append(label)
        self.template_list = list(self.template2label.keys())
        self.template2id = {template : i for i, template in enumerate(self.template_list)}
        self.label2templateid = defaultdict(list)
        for label, templates in self.templates.items():
            self.label2templateid[label].extend(self.template2id[template] for template in templates)
        
        ##construct NLI format

        NLI_label2id = {"entailment": 2, "neutral": 1, "contradiction": 0}
        self.NLI_data = []

        for item in self.data:
            ##positive example
            if item['label']!="no_relation":
                ##entailment instance
                template_id_list = self.label2templateid[item['label']]
                for tid in template_id_list:
                    premise = item['context']
                    t = self.template_list[tid]
                    hypothesis = f"{t.format(subj = item['subj'], obj = item['obj'])}."
                    label = NLI_label2id['entailment']
                    self.NLI_data.append({'premise' : premise, 'hypothesis' : hypothesis, 'label' : label})
                ##neutral instance
                neg_template_id_list = [x for x in [i for i in range(len(self.template_list))] if x not in template_id_list]
                choice_id = random.choice(neg_template_id_list)
                t = self.template_list[choice_id]
                hypothesis = f"{t.format(subj = item['subj'], obj = item['obj'])}."
                label = NLI_label2id['neutral']
                self.NLI_data.append({'premise' : premise, 'hypothesis' : hypothesis, 'label' : label})
            ##negative example
            else:
                choice_id = random.choice([i for i in range(len(self.template_list))])
                premise = item['context']
                t = self.template_list[choice_id]
                hypothesis = f"{t.format(subj = item['subj'], obj = item['obj'])}."
                label = NLI_label2id['contradiction']
                self.NLI_data.append({'premise' : premise, 'hypothesis' : hypothesis, 'label' : label})
        
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.NLI_label = [item['label'] for item in self.NLI_data]
        print('total TACRED train dataset size : ', len(self.data))
        print('total NLI dataset size : ', len(self.NLI_data))
        print(f'NLI label distribution \n entailment : {self.NLI_label.count(2)} \n neutral : {self.NLI_label.count(1)} \n contradiction : {self.NLI_label.count(0)}')

    def __len__(self):
        return len(self.NLI_data)
        
    def __getitem__(self, idx):
        s1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.NLI_data[idx]['premise']))
        s2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.NLI_data[idx]['hypothesis']))
        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        s = cls + s1 + sep + s2
        return {'input_ids' : s, 'labels' : self.NLI_data[idx]['label']}

class RE2NLI_test_dataset(Dataset):
    def __init__(self, test_file_path):
        super().__init__()
        self.file_path = test_file_path
        self.data = []
        with open(self.file_path, 'r') as f:
            for i, line in tqdm(enumerate(json.load(f))):
                subj = " ".join(line["token"][line["subj_start"] : line["subj_end"] + 1]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
                obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
                inst_type=f"{line['subj_type']}:{line['obj_type']}"
                context=" ".join(line["token"]).replace("-LRB-", "(").replace("-RRB-", ")").replace("-LSB-", "[").replace("-RSB-", "]")
                label=line["relation"]
                self.data.append({'subj' : subj, 'obj' : obj, 'inst_type' : inst_type, 'context' : context, 'label' : label})
        
        with open("./LV_template.json", 'r') as f:
            LV_template = json.load(f)
        self.labels = LV_template['labels']
        self.templates = LV_template['templates']
        self.valid_conditions = LV_template['valid_conditions']
        self.negative_label_id = 0

        self.label2id = {label : i for i, label in enumerate(self.labels)}
        self.n_labels = len(self.labels)
        self.template2label = defaultdict(list)
        for label, templates in self.templates.items():
            for template in templates:
                self.template2label[template].append(label)
        self.template_list = list(self.template2label.keys())
        self.template2id = {template : i for i, template in enumerate(self.template_list)}
        self.label2templateid = defaultdict(list)
        for label, templates in self.templates.items():
            self.label2templateid[label].extend(self.template2id[template] for template in templates)
        
        ##valid_entity_type
        self._valid_conditions = {}
        self._always_valid_labels = np.zeros(self.n_labels)
        self._always_valid_labels[self.negative_label_id] = 1.0
        for label, conditions in self.valid_conditions.items():
            if label not in self.labels:
                continue
            for condition in conditions:
                if condition == "*":
                    self._always_valid_labels[self.label2id[label]] = 1.0
                    continue
                if condition not in self._valid_conditions:
                    self._valid_conditions[condition] = np.zeros(self.n_labels)
                    if self.negative_label_id>=0:
                        self._valid_conditions[condition][self.negative_label_id] = 1.0
                self._valid_conditions[condition][self.label2id[label]] = 1.0
        
        self.NLI_data = [] ## template 개수 72 x data instance 개수 ?
        for item in self.data:
            premise = item['context']
            for template in self.template_list:
                hypothesis = f"{template.format(subj = item['subj'], obj = item['obj'])}."
                self.NLI_data.append({'premise' : premise, 'hypothesis' : hypothesis, 'label' : self.label2id[item['label']], 'inst_type' : item['inst_type']})
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        print('total NLI dataset size : ', len(self.NLI_data))
    def __len__(self):
        return len(self.NLI_data)
    
    def __getitem__(self, idx):
        s1 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.NLI_data[idx]['premise']))
        s2 = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.NLI_data[idx]['hypothesis']))
        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        s = cls + s1 + sep + s2
        return {'input_ids' : s, 'labels' : self.NLI_data[idx]['label'], 'inst_type' : self.NLI_data[idx]['inst_type']}
