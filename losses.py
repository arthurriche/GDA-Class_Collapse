import torch
import torch.nn as nn

class ContrastiveLossWithAlpha(nn.Module):
    def __init__(self, temperature=0.5, alpha=0.5):
        super(ContrastiveLossWithAlpha, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.matmul(representations, representations.T) / self.temperature

        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z_i.device)

        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z_i.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z_i.device)

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss