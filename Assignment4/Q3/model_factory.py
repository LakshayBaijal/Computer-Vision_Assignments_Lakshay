import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import os

class MLP(nn.Module):
    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class ClipCapModel(nn.Module):
    def __init__(self, prefix_length, prefix_size=512, use_mlp=True):
        super(ClipCapModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        
        if use_mlp:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, 
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)

    def forward(self, tokens, prefix):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        return self.gpt(inputs_embeds=embedding_cat)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_clipcap(use_mlp, checkpoint_path, prefix_length=10, fine_tune_gpt=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClipCapModel(prefix_length=prefix_length, use_mlp=use_mlp).to(device)
    
    # Task 3.3/3.1 Logic: If we aren't fine-tuning, freeze GPT-2
    if not fine_tune_gpt:
        for param in model.gpt.parameters():
            param.requires_grad = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    
    return model, device