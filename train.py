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








