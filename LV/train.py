import os
import torch
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
import numpy as np
from dataset import RE2NLI_dataset
import random

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
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = {'input_ids': input_ids, 'attention_mask' : input_mask, 'labels' : labels}
    return output

def main(seed = 42):
    seed_everything(seed)
    wandb.init(project='LV')
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = RE2NLI_dataset()
    config = AutoConfig.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli", config = config)
    model.to(device)
    training_args = TrainingArguments(
        do_train=True,
        output_dir = f'../../LV_result{seed}/',
        evaluation_strategy='no',
        per_device_train_batch_size=32,
        learning_rate=4e-6,
        warmup_steps=1000,
        num_train_epochs=2.0,
        save_strategy='no',
        seed = 0,
        report_to='wandb',
        run_name = 'LV_training',
        logging_strategy='steps',
        logging_steps = 100,
        fp16=True
    )
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset=train_dataset,
        tokenizer=AutoTokenizer.from_pretrained("roberta-large-mnli"),
        data_collator=collate_fn
    )
    trainer.train()
    trainer.save_model()



if __name__ == "__main__":
    main(0)