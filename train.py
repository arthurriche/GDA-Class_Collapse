import torch
import torch.nn as nn
import torch.optim as optim
from dataset import create_house_dataset
from losses import ContrastiveLossWithAlpha
from visualization import plot_and_save

def train_model(alpha=0.5, num_epochs=10):
    model = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 64))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = ContrastiveLossWithAlpha(alpha=alpha)

    data, labels = create_house_dataset()
    data = data.to(torch.device('cpu'))
    labels = labels.to(torch.device('cpu'))

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z_i = model(data)
        z_j = model(data)
        loss = criterion(z_i, z_j)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model, data, labels

if __name__ == "__main__":
    alphas = [0.0, 0.25, 0.5, 0.75, 1]

    for alpha in alphas:
        model, data, labels = train_model(alpha=alpha)

        # Visualization
        with torch.no_grad():
            representations = model(data).numpy()

        plot_and_save(data.numpy(), labels.numpy(), representations, alpha)