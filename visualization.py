import matplotlib.pyplot as plt
import os

def plot_and_save(data, labels, representations, alpha, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f'Original Data (alpha={alpha})')
    plt.savefig(os.path.join(output_dir, f'original_data_alpha_{alpha}.png'))

    plt.subplot(1, 2, 2)
    plt.scatter(representations[:, 0], representations[:, 1], c=labels, cmap='viridis', s=5)
    plt.title(f'Learned Representations (alpha={alpha})')
    plt.savefig(os.path.join(output_dir, f'learned_representations_alpha_{alpha}.png'))