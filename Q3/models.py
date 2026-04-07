from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, input_dim: int = 3, emb_dims: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.conv4 = nn.Conv1d(256, emb_dims, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(emb_dims)

        self.fc1 = nn.Linear(emb_dims, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor, return_details: bool = False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        pre_pool = F.relu(self.bn4(self.conv4(x)))

        global_feature, critical_indices = torch.max(pre_pool, dim=2)

        cls = F.relu(self.bn5(self.fc1(global_feature)))
        cls = F.relu(self.bn6(self.fc2(cls)))
        cls = self.dropout(cls)
        logits = self.fc3(cls)

        if return_details:
            details: Dict[str, torch.Tensor] = {
                "pre_pool": pre_pool,
                "critical_indices": critical_indices,
                "global_feature": global_feature,
            }
            return logits, details
        return logits
