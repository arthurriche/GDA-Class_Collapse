import numpy as np
import torch

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

def create_house_dataset(num_strata=3, num_samples_per_stratum=100):
    data = []
    labels = []
    
    # Define strata based on price ranges
    price_ranges = [(100000, 200000), (200000, 400000), (400000, 800000)]
    
    for stratum_id in range(num_strata):
        price_min, price_max = price_ranges[stratum_id]
        
        # Generate random data for each feature
        num_rooms = np.random.randint(1, 10, num_samples_per_stratum)
        square_footage = np.random.randint(500, 5000, num_samples_per_stratum)
        age = np.random.randint(0, 100, num_samples_per_stratum)
        price = np.random.randint(price_min, price_max, num_samples_per_stratum)
        
        # Combine features into a single array
        stratum_data = np.stack([num_rooms, square_footage, age, price], axis=1)
        data.append(stratum_data)
        
        # Label each stratum with its ID
        labels += [stratum_id] * num_samples_per_stratum
    
    data = np.concatenate(data)
    print("Dataset created with shape:", data.shape)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels)

# Example usage
data, labels = create_house_dataset()
print(data[:5])  # Print first 5 samples
print(labels[:5])  # Print first 5 labels