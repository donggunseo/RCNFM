
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import os
from tqdm import tqdm
import random

LABEL_TO_ID = {
	"no_relation": 0,
	"per:title": 1,
	"org:top_members/employees": 2,
	"per:employee_of": 3,
	"org:alternate_names": 4,
	"org:country_of_headquarters": 5,
	"per:countries_of_residence": 6,
	"org:city_of_headquarters": 7,
	"per:cities_of_residence": 8,
	"per:age": 9,
	"per:stateorprovinces_of_residence": 10,
	"per:origin": 11,
	"org:subsidiaries": 12,
	"org:parents": 13,
	"per:spouse": 14,
	"org:stateorprovince_of_headquarters": 15,
	"per:children": 16,
	"per:other_family": 17,
	"per:alternate_names": 18,
	"org:members": 19,
	"per:siblings": 20,
	"per:schools_attended": 21,
	"per:parents": 22,
	"per:date_of_death": 23,
	"org:member_of": 24,
	"org:founded_by": 25,
	"org:website": 26,
	"per:cause_of_death": 27,
	"org:political/religious_affiliation": 28,
	"org:founded": 29,
	"per:city_of_death": 30,
	"org:shareholders": 31,
	"org:number_of_employees/members": 32,
	"per:date_of_birth": 33,
	"per:city_of_birth": 34,
	"per:charges": 35,
	"per:stateorprovince_of_death": 36,
	"per:religion": 37,
	"per:stateorprovince_of_birth": 38,
	"per:country_of_birth": 39,
	"org:dissolved": 40,
	"per:country_of_death": 41
}

VALID_RELATION_FOR_PAIR ={
    ('ORGANIZATION', 'PERSON'): ['org:founded_by', 'org:top_members/employees', 'org:shareholders'],
    ('ORGANIZATION', 'ORGANIZATION'): ['org:alternate_names', 'org:member_of', 'org:members', 'org:parents', 'org:subsidiaries', 'org:shareholders'],
    ('ORGANIZATION', 'NUMBER'): ['org:number_of_employees/members'],
    ('ORGANIZATION', 'DATE'): ['org:dissolved', 'org:founded'],
    ('ORGANIZATION', 'LOCATION'): ['org:member_of', 'org:city_of_headquarters', 'org:country_of_headquarters', 'org:stateorprovince_of_headquarters', 'org:parents', 'org:subsidiaries'],
    ('ORGANIZATION', 'CITY'): ['org:city_of_headquarters'],
    ('ORGANIZATION', 'MISC'): ['org:alternate_names'],
    ('ORGANIZATION', 'COUNTRY'): ['org:member_of', 'org:members', 'org:country_of_headquarters', 'org:parents', 'org:subsidiaries'],
    ('ORGANIZATION', 'RELIGION'): ['org:political/religious_affiliation'],
    ('ORGANIZATION', 'URL'): ['org:website'],
    ('ORGANIZATION', 'STATE_OR_PROVINCE'): ['org:member_of', 'org:stateorprovince_of_headquarters', 'org:parents'],
    ('ORGANIZATION', 'IDEOLOGY'): ['org:political/religious_affiliation'],
    ('PERSON', 'PERSON'): ['per:children', 'per:siblings', 'per:spouse', 'per:other_family', 'per:alternate_names', 'per:parents'],
    ('PERSON', 'ORGANIZATION'): ['per:employee_of', 'per:schools_attended'],
    ('PERSON', 'NUMBER'): ['per:age'],
    ('PERSON', 'DATE'): ['per:date_of_death', 'per:date_of_birth'],
    ('PERSON', 'NATIONALITY'): ['per:countries_of_residence', 'per:origin', 'per:country_of_birth', 'per:country_of_death'],
    ('PERSON', 'LOCATION'): ['per:employee_of', 'per:cities_of_residence', 'per:stateorprovinces_of_residence', 'per:countries_of_residence', 'per:origin', 'per:city_of_death', 'per:stateorprovince_of_death', 'per:country_of_birth', 'per:city_of_birth', 'per:country_of_death'],
    ('PERSON', 'TITLE'): ['per:title'],
    ('PERSON', 'CITY'): ['per:cities_of_residence', 'per:city_of_death', 'per:city_of_birth'],
    ('PERSON', 'MISC'): ['per:alternate_names'],
    ('PERSON', 'COUNTRY'): ['per:countries_of_residence', 'per:origin', 'per:country_of_birth', 'per:country_of_death'],
    ('PERSON', 'CRIMINAL_CHARGE'): ['per:charges'],
    ('PERSON', 'RELIGION'): ['per:religion'],
    ('PERSON', 'DURATION'): ['per:age'],
    ('PERSON', 'STATE_OR_PROVINCE'): ['per:stateorprovinces_of_residence', 'per:stateorprovince_of_birth', 'per:stateorprovince_of_death'],
    ('PERSON', 'CAUSE_OF_DEATH'): ['per:cause_of_death']
}

RELATION_SENTENCE = {
    'org:founded_by': ['subj', 'is', 'founded', 'by', 'obj', '.'],
    'org:top_members/employees': ['obj', 'is', 'the', 'top', 'member', 'or', 'employee', 'of', 'subj', '.'],
    'org:shareholders': ['The', 'shareholder', 'of', 'subj', 'is', 'obj', '.'],
    'org:alternate_names': ['The', 'alternate', 'name', 'of', 'subj', 'is', 'obj', '.'],
    'org:member_of': ['subj', 'is', 'the', 'member', 'of', 'obj', '.'],
    'org:members': ['obj', 'is', 'the', 'member', 'of', 'subj', '.'],
    'org:parents': ['The', 'parent', 'of', 'subj', 'is', 'obj', '.'],
    'org:subsidiaries': ['The', 'subsidiary', 'of', 'subj', 'is', 'obj', '.'],
    'org:number_of_employees/members': ['The','number', 'of', 'employees', 'or', 'members', 'of', 'subj', 'are', 'obj', '.'],
    'org:dissolved': ['subj', 'is', 'dissolved', 'in', 'obj', '.'],
    'org:founded': ['subj', 'is', 'founded', 'in', 'obj', '.'],
    'org:city_of_headquarters': ['The', 'headquarter', 'of', 'subj', 'is', 'located', 'in', 'city', 'obj', '.'],
    'org:country_of_headquarters': ['The', 'headquarter', 'of', 'subj', 'is', 'located', 'in', 'country', 'obj', '.'],
    'org:stateorprovince_of_headquarters': ['The', 'headquarter', 'of', 'subj', 'is', 'located', 'in', 'state', 'or', 'province', 'obj', '.'],
    'org:political/religious_affiliation': ['The', 'political', 'affiliation', 'of', 'subj', 'is', 'obj', '.'],
    'org:website': ['The', 'website', 'address', 'of', 'subj', 'is', 'obj', '.'],
    'per:children': ['The', 'children', 'of', 'subj', 'is', 'obj', '.'],
    'per:siblings': ['The', 'sibling', 'of', 'subj', 'is', 'obj', '.'],
    'per:spouse': ['The', 'spouse', 'of', 'subj', 'is', 'obj', '.'],
    'per:other_family': ['The', 'other', 'familiy', 'of', 'subj', 'except', 'children', ',' , 'sibling' , ',','spouse', 'and', 'parents', 'is', 'obj', '.'],
    'per:alternate_names': ['The', 'alternate', 'name', 'of', 'subj', 'is', 'obj', '.'],
    'per:parents': ['The', 'parent', 'of', 'subj', 'is', 'obj', '.'],
    'per:employee_of': ['subj', 'is', 'employee', 'of', 'obj', '.'],
    'per:schools_attended': ['The', 'school', 'subj', 'attended', 'is', 'obj', '.'],
    'per:age': ['The', 'age', 'of', 'subj', 'is', 'obj', '.'],
    'per:date_of_death': ['subj', 'died', 'in', 'obj', '.'],
    'per:date_of_birth': ['subj', 'was', 'born', 'in', 'obj', '.'],
    'per:countries_of_residence': ['The', 'residence', 'of', 'subj', 'is', 'located', 'in', 'country', 'obj', '.'],
    'per:origin': ['The', 'origin', 'of', 'subj', 'is', 'obj', '.'],
    'per:country_of_birth': ['subj', 'was', 'born', 'in', 'country', 'obj', '.'],
    'per:country_of_death': ['subj', 'died', 'in', 'country', 'obj', '.'],
    'per:cities_of_residence': ['The', 'residence', 'of', 'subj', 'is', 'located', 'in', 'city', 'obj', '.'],
    'per:stateorprovinces_of_residence': ['The', 'residence', 'of', 'subj', 'is', 'located', 'in','state', 'or', 'province', 'obj', '.'],
    'per:city_of_death': ['subj', 'died', 'in', 'city', 'obj', '.'],
    'per:stateorprovince_of_death': ['subj', 'died', 'in', 'state', 'or', 'province', 'obj', '.'],
    'per:city_of_birth': ['subj', 'was', 'born', 'in', 'city', 'obj', '.'],
    'per:title': ['The', 'title', 'of', 'subj', 'is', 'obj', '.'],
    'per:charges': ['The', 'crime', 'charge', 'of', 'subj', 'is', 'obj', '.'],
    'per:religion': ['The', 'religion', 'of', 'subj', 'is', 'obj', '.'],
    'per:stateorprovince_of_birth': ['subj', 'was', 'born', 'in', 'state', 'or', 'province', 'obj', '.'],
    'per:cause_of_death': ['subj', 'died', 'because', 'of', 'obj', '.']
}

class RE_dataset(Dataset):
    def __init__(self, args, do_eval=False, do_test=False):
        if not do_eval:
            file_path = os.path.join(args.data_dir, 'train.json')
        elif not do_test:
            file_path = os.path.join(args.data_dir, 'dev.json')
        else:
            file_path = os.path.join(args.data_dir, 'test.json')
        assert os.path.isfile(file_path)
        self.args = args
        self.file_path = file_path
        self.evaluate = do_eval
        self.test = do_test
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.data = []
        self.label = []
        self.id = []
        self.label_ori = []
        self.answer = []
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
        for d in tqdm(data, desc = 'dataset preprocessing'):
            label = LABEL_TO_ID[d['relation']]
            if self.args.except_no_relation and label==0 and (not self.evaluate):
                continue
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
                if self.args.remove_marker:
                    if i == ss:
                        t = ['@'] + ['*'] + subj_type + ['*'] + t
                    if i == se:
                        t = t + ['@']
                    if i == os:
                        t = ["#"] + ['^'] + obj_type + ['^'] + t
                    if i == oe:
                        t = t + ["#"]
                sentence.extend(t)
            subj_entity = ['@'] + ['*'] + subj_type + ['*'] if self.args.remove_marker else []
            obj_entity = ["#"] + ['^'] + obj_type + ['^'] if self.args.remove_marker else []
            for j, token in enumerate(tokens[ss:se+1]):
                t = self.tokenizer.tokenize(token)
                subj_entity.extend(t)
            if self.args.remove_marker:
                subj_entity.append('@') 
            for j, token in enumerate(tokens[os:oe+1]):
                t = self.tokenizer.tokenize(token)
                obj_entity.extend(t)
            if self.args.remove_marker:
                obj_entity.append('#') 

            candidate_relations_list = VALID_RELATION_FOR_PAIR[(subj_type_ori, obj_type_ori)].copy() if self.args.no_restriction else list(LABEL_TO_ID.keys())[1:]
            if self.args.zero_threshold and (not self.evaluate):
                if d['relation']!='no_relation' and len(candidate_relations_list)>self.args.zero_threshold_num:
                    temp = [d['relation']]
                    candidate_relations_list.remove(d['relation'])
                    temp.extend(random.sample(candidate_relations_list, self.args.zero_threshold_num-1))
                    candidate_relations_list = temp
                elif d['relation']=='no_relation' and len(candidate_relations_list)>self.args.zero_threshold_num:
                    temp = random.sample(candidate_relations_list, self.args.zero_threshold_num)
                    candidate_relations_list = temp
            binary_label_list= [0 if LABEL_TO_ID[k]!=label else 1 for k in candidate_relations_list]
            sentence2 = []
            for candidate_relation in candidate_relations_list:
                relation_sentence = RELATION_SENTENCE[candidate_relation]
                subj_idx = relation_sentence.index('subj')
                obj_idx = relation_sentence.index('obj')
                tokenized_relation_sentence = []
                for i, token in enumerate(relation_sentence):
                    if i==subj_idx:
                        tokenized_relation_sentence.extend(subj_entity)
                    elif i==obj_idx:
                        tokenized_relation_sentence.extend(obj_entity)
                    else:
                        t = self.tokenizer.tokenize(token)
                        tokenized_relation_sentence.extend(t)
                sentence2.append(tokenized_relation_sentence)
            s = [(sentence, r) for r in sentence2]
            self.data.extend(s)
            self.label.extend(binary_label_list)
            self.id.extend([id for _ in range(len(candidate_relations_list))])
            self.answer.append(label)
            self.label_ori.extend([LABEL_TO_ID[r] for r in candidate_relations_list])
        print('total dataset size : ',  len(self.data))
        print(f'class distribution \n 0 : {self.label.count(0)} \n 1 : {self.label.count(1)}')
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        s1 = self.tokenizer.convert_tokens_to_ids(self.data[idx][0])
        s2 = self.tokenizer.convert_tokens_to_ids(self.data[idx][1])
        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        s = cls + s1 + sep + s2
        return {'input_ids': s, 'labels' : self.label[idx], 'id' : self.id[idx], 'label_ori' : self.label_ori[idx]}


