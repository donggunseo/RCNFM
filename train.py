from cgi import test
import os
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import f1_score, accuracy_score
import argparse
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils import set_seed, collate_fn
from tqdm import tqdm
import os
import numpy as np
from model import RECSE_Model
from torch.cuda.amp import GradScaler
from dataset import RE_dataset
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
def train(model, args, train_dataset, eval_dataset, test_dataset):
    dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last = True)
    total_steps = int(len(dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))
    

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(dataloader)):
            model.train()
            label = batch[2].to(args.device)
            inputs = {'input_ids': batch[0].to(args.device), 'attention_mask': batch[1].to(args.device)}
            outputs = model(**inputs)
            loss = criterion(outputs, label)
            loss = loss/args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)
            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                evaluate(model, args, eval_dataset, num_steps)
                evaluate(model, args, test_dataset, num_steps, False)

def evaluate(model, args, test_dataset, num_steps, flag=True):
    dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last = False)
    answer = test_dataset.answer
    logits = []
    preds = []
    ids = []
    labels = []
    label_oris = []
    for i, batch in enumerate(tqdm(dataloader)):
        model.eval()
        label = batch[2]
        id = batch[3]
        label_ori = batch[4]
        inputs = {'input_ids': batch[0].to(args.device), 'attention_mask': batch[1].to(args.device)}
        with torch.no_grad():
            logit = model(**inputs)
            pred = torch.argmax(logit, dim=-1)
        logits+=logit.tolist()
        preds+=pred.tolist()
        ids+=id
        labels+=label.tolist()
        label_oris+=label_ori
    data_edge = [0]
    final_pred = []
    for i in range(len(ids)-1):
        if ids[i]!=ids[i+1]:
            data_edge.append(i+1)
    data_edge.append(len(ids))
    for i in range(len(data_edge)-1):
        cur_pred = preds[data_edge[i]:data_edge[i+1]]
        cur_logit = logits[data_edge[i]:data_edge[i+1]]
        cur_labels = labels[data_edge[i]:data_edge[i+1]]
        cur_ori = label_oris[data_edge[i]:data_edge[i+1]]
        temp_pred = []
        for i in range(len(cur_pred)):
            if cur_pred[i]==1:
                temp_pred.append((cur_logit[i][1], cur_ori[i]))
        if len(temp_pred)==0:
            final_pred.append(0)
        elif len(temp_pred)==1:
            final_pred.append(temp_pred[0][1])
        else:
            temp_pred = sorted(temp_pred, key = lambda x : x[0], reverse=True)
            final_pred.append(temp_pred[0][1])
    f1 = f1_score(answer, final_pred, average = 'micro')
    print(f"f1 score : {f1}")
    if flag:
        wandb.log({'eval f1' : f1}, step = num_steps)
    else:
        wandb.log({'test f1' : f1}, step = num_steps)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default="../data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli', type=str)

    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=42)
    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RECSE")
    parser.add_argument("--run_name", type=str, default="tacred")

    args = parser.parse_args()
    wandb.init(project=args.project_name, name=args.run_name)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)
    args.n_gpu = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    config.gradient_checkpointing = True
    model = RECSE_Model(args, config)
    print(args.n_gpu)
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids = list(range(args.n_gpu)))
    model.to(args.device)
    set_seed(args)
    train_dataset = RE_dataset(args)
    eval_dataset = RE_dataset(args, do_eval = True)
    test_dataset = RE_dataset(args, do_eval = True, do_test = True)

    train(model, args, train_dataset, eval_dataset, test_dataset)

    evaluate(model, args, test_dataset, 0, False)

if __name__ == "__main__":
    main()









