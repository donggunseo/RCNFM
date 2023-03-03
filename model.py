import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast


class RECSE_Model(nn.Module):
    def __init__(self, args=None, config=None):
        super().__init__()
        self.model_name_or_path= args.model_name_or_path
        self.model = AutoModel.from_pretrained(self.model_name_or_path, config=config)
        self.classifier = nn.Linear(1024, 2)
        self.dropout = nn.Dropout(args.dropout_prob)
        self.args = args
    
    @autocast()
    def forward(self, input_ids=None, attention_mask=None):
        output = self.model(input_ids, attention_mask = attention_mask)
        if self.args.pooler:
            pooled_cls_output = output[1]
        else:
            pooled_cls_output = output[0][:,0]
        pooled_cls_output = self.dropout(pooled_cls_output)
        cls_output = self.classifier(pooled_cls_output)
        return cls_output