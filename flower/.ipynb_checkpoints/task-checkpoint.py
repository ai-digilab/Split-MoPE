import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

class ClientBackbone(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
        )
        # Freeze parameters, i.e. non-trainable
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.fc(x)

class ServerHead(nn.Module):
    def __init__(self, num_clients, client_embedding_dim=32, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(client_embedding_dim * num_clients, client_embedding_dim * num_clients * 2),
            nn.ReLU(),
            nn.Linear(client_embedding_dim * num_clients * 2, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

class CustomDataset(Dataset):
    def __init__(self, base_dataset, batch_size, num_clients, p_miss):
        self.dataset = base_dataset
        num_samples = len(base_dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(
            num_batches, num_clients, p_miss
        )
        self.batch_size = batch_size
        self.classes = getattr(base_dataset, 'classes', None)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        data = self.dataset[idx]

        features, label = data[:-1], data[-1]

        mask = self.batch_patterns[batch_idx]
        return (*features, label, torch.tensor(mask, dtype=torch.bool))

def collate_fn_text(batch):
    """Collate function for text-based datasets like breast cancer."""
    *features, labels, masks = zip(*batch)
    texts = list(features[0])  
    labels = torch.tensor(labels)
    mask = masks[0]
    return (texts, labels, mask)

def collate_fn(batch):
    *features, labels, masks = zip(*batch)
    stacked_features = [torch.stack(feature_set) for feature_set in features]
    labels = torch.tensor(labels)
    mask = masks[0] 
    return (*stacked_features, labels, mask)

def _get_mask_per_batch(num_batches, num_blocks, p_miss=None):
    """
    Generate possible alignments and take as many as the amount of batches based on the probability of each one.

    We subtract 1 as in LASER-VFL the figure of active participant does not exist. For us there is a single active, whose block is always visible
    """
    num_blocks -= 1
    p_observed = None if p_miss is None else (1 - p_miss)
    if isinstance(p_observed, float):
        p_observed_array = np.array([p_observed] * num_blocks)
    elif p_observed is None:
        p_observed_array = np.random.beta(2.0, 2.0, num_blocks)

    patterns = [
        np.array([bool(int(x)) for x in bin(i)[2:].zfill(num_blocks)])
        for i in range(2 ** (num_blocks))
    ]

    probabilities = []
    for pattern in patterns:
        prob = 1.0
        for block, p_observed in zip(pattern, p_observed_array):
            prob *= p_observed if block else (1 - p_observed)
        probabilities.append(prob)

    chosen_patterns = np.random.choice(
        len(patterns),
        size=num_batches,
        p=np.array(probabilities) / np.sum(probabilities),
    )
    batch_patterns = [patterns[i] for i in chosen_patterns]

    return batch_patterns, p_observed_array

def load_sklearn_data(partition_id: int, num_partitions: int, p_miss_train=0.5, p_miss_test=0.5, batch_size=32):
    """Load Sklearn data and split features vertically with VFL masking."""
    
    # 1. Load the full dataset
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int64)

    # 2. Split into Train and Test (SHUFFLE=FALSE is critical for VFL alignment)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # 3. VERTICAL SPLIT
    total_features = X_train.shape[1]
    features_per_client = total_features // num_partitions
    
    start_idx = partition_id * features_per_client
    end_idx = total_features if partition_id == num_partitions - 1 else (partition_id + 1) * features_per_client

    X_train_part = X_train[:, start_idx:end_idx]
    X_test_part = X_test[:, start_idx:end_idx]

    # 4. Create the Base Datasets
    train_base = TensorDataset(torch.from_numpy(X_train_part), torch.from_numpy(y_train))
    test_base = TensorDataset(torch.from_numpy(X_test_part), torch.from_numpy(y_test))

    # 5. Wrap in CustomDataset to apply LASER-VFL masking
    batch_size = batch_size
    
    train_wrapped = CustomDataset(
        base_dataset=train_base, 
        batch_size=batch_size, 
        num_clients=num_partitions, 
        p_miss=p_miss_train
    )
    
    test_wrapped = CustomDataset(
        base_dataset=test_base, 
        batch_size=batch_size, 
        num_clients=num_partitions, 
        p_miss=p_miss_test
    )

    train_loader = DataLoader(
        train_wrapped, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_wrapped, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    return train_loader, test_loader