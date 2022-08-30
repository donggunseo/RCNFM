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

def train(model, args, train_dataset):
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
            id = batch[3]
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

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", default="./data/tacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    config.gradient_checkpointing = True
    model = RECSE_Model(args, config)
    if args.n_gpu > 1:
        model = nn.DataParallel(model, device_ids = list(range(args.n_gpu)))
    model.to(device)

    train_dataset = RE_dataset(args)
    eval_dataset = RE_dataset(args, do_eval = True)
    test_dataset = RE_dataset(args, do_eval = True, do_test = True)

    train(model, args, train_dataset)

if __name__ == "__main__":
    main()









