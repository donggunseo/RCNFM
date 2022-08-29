import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast


class RECSE_Model(nn.Module):
    def __init__(self, args=None, config=None):
        self.model_type = args.model_type
        self.model = AutoModel.from_pretrained(self.model_type, config=config)
        self.classifier = nn.Linear(1024, 2)
    
    @autocast
    def forward(self, input_ids=None, attention_mask=None):
        output = self.model(input_ids, attention_mask = attention_mask)
        pooled_cls_output = output[0][:,0]
        cls_output = self.classifier(pooled_cls_output)
        return cls_output