from pathlib import Path
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from torchvision import datasets, transforms
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tinyimagenet import TinyImageNet

from .custom_dataset import (
    CustomDataset,
    collate_fn,
    collate_fn_text,
    BreastCancerDataset,
)


def get_dataloaders(args, config, p_miss_test=0.0):
    # Extract configuration parameters
    dataset_name = config["dataset"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    # Set up data directory path
    data_dir = "./cache"

    # Get image transformations for the specified dataset
    if dataset_name not in ["breast_cancer", "uci_credit_default", "npha"]:
        transform = get_image_transforms(dataset_name)

    # Select the appropriate dataset class based on dataset name
    if dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
        # Create training dataset
        train_set = dataset_class(
            root=data_dir, download=True, train=True, transform=transform
        )
    
        # Create test dataset
        test_set = dataset_class(
            root=data_dir, download=True, train=False, transform=transform
        )
    elif dataset_name == "cifar100":
        dataset_class = datasets.CIFAR100
        # Create training dataset
        train_set = dataset_class(
            root=data_dir, download=True, train=True, transform=transform
        )
    
        # Create test dataset
        test_set = dataset_class(
            root=data_dir, download=True, train=False, transform=transform
        )
    elif dataset_name == "breast_cancer":
        bc = load_breast_cancer()
        
        # Get feature names from dataset
        feature_names = bc.feature_names.tolist()
                
        X = bc.data.astype(np.float32)
        y = bc.target.astype(np.int64)
        
        # -----------------------------
        # Permute columns so seed-642 subset is last 15
        # -----------------------------
        n_features = X.shape[1]
        
        # Generate ONLY the specific random subset for seed 642
        random_seed = 642
        np.random.seed(random_seed)
        target_idxs = sorted(np.random.choice(n_features, size=15, replace=False).tolist())
        
        # Build permutation: all non-target columns first (keep original order), then target columns
        non_target = [i for i in range(n_features) if i not in target_idxs]
        perm = non_target + target_idxs  # length 30
        
        # Apply permutation to X and feature_names
        X = X[:, perm]
        feature_names = [feature_names[i] for i in perm]
        
        # for j in range(n_features - 15, n_features):
        #     print(j, feature_names[j])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        train_set = BreastCancerDataset(X_train, y_train, feature_names=feature_names)
        test_set = BreastCancerDataset(X_test, y_test, feature_names=feature_names)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Supported datasets are 'cifar10' and 'cifar100'."
        )

    # use_text = (dataset_name = "breast_cancer")
    use_text = False
    # Create data loaders for training and testing
    train_loader = create_data_loader(
        base_dataset=train_set,
        batch_size=batch_size,
        num_clients=args.num_clients,
        p_miss=args.p_miss_train,
        num_workers=num_workers,
        use_text_collate=use_text,
    )

    test_loader = create_data_loader(
        base_dataset=test_set,
        batch_size=batch_size,
        num_clients=args.num_clients,
        p_miss=args.p_miss_test,
        num_workers=num_workers,
        use_text_collate=use_text,
    )

    return train_loader, test_loader

def block_to_text(block_tensor, feature_names, client_idx, num_clients):
    """Convert a numerical block to text representation."""
    # Calculate which features this block contains
    num_features = len(feature_names)
    features_per_client = num_features // num_clients
    
    start_idx = client_idx * features_per_client
    end_idx = (client_idx + 1) * features_per_client
    
    block_feature_names = feature_names[start_idx:end_idx]
    
    # Convert tensor values to text
    block_values = block_tensor.cpu().numpy() if isinstance(block_tensor, torch.Tensor) else block_tensor
    
    parts = [f"{name}: {value:.4f}" for name, value in zip(block_feature_names, block_values)]
    return " | ".join(parts)

def create_data_loader(
    base_dataset,
    batch_size,
    num_clients,
    p_miss,
    num_workers=0,
    drop_last=False,
    sampler=None,
    use_text_collate=False,  # Add this parameter
):
    wrapped_dataset = CustomDataset(base_dataset, batch_size, num_clients, p_miss)
    
    # Choose collate function based on data type
    collate_function = collate_fn_text if use_text_collate else collate_fn
    
    return DataLoader(
        wrapped_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_function,
        drop_last=drop_last,
        sampler=sampler,
    )


# THESE TRANSFORMS HAVE BEEN MODIFIED TO RESIZE IMAGES TO 224
def get_image_transforms(dataset_name):
    norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    if dataset_name == "cifar100":
        return transforms.Compose(
            [
                transforms.Resize((224, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
    elif dataset_name == "cifar10":
        return transforms.Compose(
            [
                transforms.Resize((224, 448)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
    else:
        raise ValueError(f"No transforms defined for '{dataset_name}'.")


def slice_cifar_block(x, partition):
    (row_start, row_end), (col_start, col_end) = partition
    return x[:, :, row_start:row_end, col_start:col_end]

def get_block_from_input(x, i, num_clients):
    x_ = x[0]
    block_size = x_.shape[1] // num_clients
    return x_[:, i * block_size : (i + 1) * block_size]