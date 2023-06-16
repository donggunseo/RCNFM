import os
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
import os
import numpy as np
from dataset import RE2NLI_test_dataset
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import json
from sklearn.metrics import f1_score, precision_recall_fscore_support
from typing import Callable, List

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    inst_type = [f["inst_type"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = {'input_ids': input_ids, 'attention_mask' : input_mask, 'labels' : labels, 'inst_type' : inst_type}
    return output

def f1_score_(labels, preds, n_labels=42):
    n_labels = max(labels) + 1 if n_labels is None else n_labels
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")

def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True, negative_label_id: int = 0):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, negative_label_id] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(int)
    output_[activations == 0, negative_label_id] = 1.00

    return output_.argmax(-1)

def find_optimal_threshold(
    labels,
    output,
    granularity=1000,
    metric=f1_score_,
    n_labels=42,
    negative_label_id: int = 0,
    apply_threshold_fn: Callable = apply_threshold,
):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in tqdm(thresholds):
        preds = apply_threshold_fn(output, threshold=t, negative_label_id=negative_label_id)
        values.append(metric(labels, preds, n_labels=n_labels))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]


def eval(seed=42):
    seed_everything(seed)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_dataset = RE2NLI_test_dataset("../../data/tacred/dev.json")
    label2templateid = eval_dataset.label2templateid
    labels = eval_dataset.labels
    valid_condition = eval_dataset._valid_conditions
    config = AutoConfig.from_pretrained(f"../../LV_result{seed}/")
    model = AutoModelForSequenceClassification.from_pretrained(f"../../LV_result{seed}/", config = config)
    model.to(device)
    dataloader = DataLoader(eval_dataset, batch_size = 32, shuffle=False, collate_fn=collate_fn, drop_last=False)
    softmax = nn.Softmax(dim=-1)
    preds = []
    inst_type_list = []
    answer_list = []
    for i, batch in enumerate(tqdm(dataloader)):
        model.eval()
        inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
        with torch.no_grad():
            result = model(**inputs)
            logit = softmax(result['logits'])
            pred = logit[:, 2]
            preds += pred.tolist()
        inst_type_list+=list(batch['inst_type'])
        answer_list+=list(batch['labels'])
    preds = np.array(preds)

    output = []
    final_answer_list=[]
    for i in range(0, len(preds), 72):
        temp_prob = preds[i:i+72]
        inst_type = inst_type_list[i]
        answer = answer_list[i]
        inst_probs = []
        for label in labels:
            if label!='no_relation':
                t_ids = label2templateid[label]
                label_prob_list = list(temp_prob[t_ids])
                label_prob = max(label_prob_list)
                inst_probs.append(label_prob)
            else:
                inst_probs.append(0)
        assert len(inst_probs) == 42, "wrong"
        inst_probs = np.array(inst_probs)
        mask_matrix = valid_condition.get(inst_type, np.zeros(len(labels)))
        probs = inst_probs * mask_matrix
        output.append(probs)
        final_answer_list.append(answer)
    output = np.array(output)

    best_threshold, best_dev = find_optimal_threshold(final_answer_list, output)
    print(best_threshold)
    print(best_dev)
    return best_threshold, best_dev

def test(best_threshold, seed=42):
    seed_everything(seed)
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval_dataset = RE2NLI_test_dataset("../../data/tacred/test.json")
    label2templateid = eval_dataset.label2templateid
    labels = eval_dataset.labels
    valid_condition = eval_dataset._valid_conditions
    config = AutoConfig.from_pretrained(f"../../LV_result{seed}/")
    model = AutoModelForSequenceClassification.from_pretrained(f"../../LV_result{seed}/", config = config)
    model.to(device)

    dataloader = DataLoader(eval_dataset, batch_size = 32, shuffle=False, collate_fn=collate_fn, drop_last=False)
    softmax = nn.Softmax(dim=-1)
    preds = []
    inst_type_list = []
    answer_list = []
    for i, batch in enumerate(tqdm(dataloader)):
        model.eval()
        inputs = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)}
        with torch.no_grad():
            result = model(**inputs)
            logit = softmax(result['logits'])
            pred = logit[:, 2]
            preds += pred.tolist()
        inst_type_list+=list(batch['inst_type'])
        answer_list+=list(batch['labels'])
    preds = np.array(preds)
    output = []
    final_answer_list=[]
    for i in range(0, len(preds), 72):
        temp_prob = preds[i:i+72]
        inst_type = inst_type_list[i]
        answer = answer_list[i]
        inst_probs = []
        for label in labels:
            if label!='no_relation':
                t_ids = label2templateid[label]
                label_prob_list = list(temp_prob[t_ids])
                label_prob = max(label_prob_list)
                inst_probs.append(label_prob)
            else:
                inst_probs.append(0)
        assert len(inst_probs) == 42, "wrong"
        inst_probs = np.array(inst_probs)
        mask_matrix = valid_condition.get(inst_type, np.zeros(len(labels)))
        probs = inst_probs * mask_matrix
        output.append(probs)
        final_answer_list.append(answer)
    output = np.array(output)
    output_ = apply_threshold(output, threshold=best_threshold)
    prec, rec, f1, _ = precision_recall_fscore_support(final_answer_list, output_,labels = [i for i in range(1,42)], average = 'micro')
    exp_result = {'prec' : prec, 'rec' : rec, 'f1' : f1}
    return exp_result


    
def main():
    seed = 0
    best_th, best_dev = eval(seed)
    result = test(best_th, seed)
    print(seed)
    print(result)
    result['best_th'] = best_th
    result['dev_f1'] = best_dev
    with open(f'./result_{seed}.json', 'w') as f:
        json.dump(result, f, indent=2)
if __name__ == "__main__":
    main()
