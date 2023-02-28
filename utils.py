import torch
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ids = [f["id"] for f in batch]
    label_oris = [f['label_ori'] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = (input_ids, input_mask, labels, ids, label_oris)
    return output

def get_f1(answer, pred):
    correct_by_relation = 0
    guessed_by_relation = 0
    gold_by_relation = 0
    for i in range(len(answer)):
        if answer[i]==pred[i] and pred[i]!=0:
            correct_by_relation+=1
        if pred[i]!=0:
            guessed_by_relation+=1
        if answer[i]!=0:
            gold_by_relation+=1
    prec_micro = float(correct_by_relation) / float(guessed_by_relation)
    recall_micro = float(correct_by_relation) / float(gold_by_relation)
    f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro+recall_micro)
    return f1_micro