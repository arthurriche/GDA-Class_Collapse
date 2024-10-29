import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

def create_dataset(num_classes=3, num_strata=2, num_samples=100):
    data = []
    labels = []
    for class_id in range(num_classes):
        for stratum_id in range(num_strata):
            mean = np.random.rand(2) * (class_id + 1)
            points = mean + 0.1 * np.random.randn(num_samples, 2)
            data.append(points)
            labels += [class_id] * num_samples
    data = np.concatenate(data)
    print("Dataset created with shape:", data.shape)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)

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

def train_model(alpha=0.5, num_epochs=100):
    model = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 64))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ContrastiveLossWithAlpha(alpha=alpha)

    data, labels = create_dataset()
    data = data.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z_i = model(data)
        z_j = model(data)
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model, data, labels

alphas = [0.0, 0.25, 0.5, 0.75, 1]

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)

for alpha in alphas:
    model, data, labels = train_model(alpha=alpha)

    # Visualization
    with torch.no_grad():
        representations = model(data).numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f'Original Data (alpha={alpha})')
    plt.savefig(f'outputs/original_data_alpha_{alpha}.png')

    plt.subplot(1, 2, 2)
    plt.scatter(representations[:, 0], representations[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f'Learned Representations (alpha={alpha})')
    plt.savefig(f'outputs/learned_representations_alpha_{alpha}.png')

    plt.show()