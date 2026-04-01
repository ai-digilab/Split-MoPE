# Copied and adapted from LASER-VFL: https://github.com/Valdeira/LASER-VFL/tree/master

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import os
import torch
import pandas as pd


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


def collate_fn(batch):
    *features, labels, masks = zip(*batch)
    stacked_features = [torch.stack(feature_set) for feature_set in features]
    labels = torch.tensor(labels)
    mask = masks[0]  # use the first mask (assuming same for all in the batch)
    return (*stacked_features, labels, mask)

def collate_fn_text(batch):
    """Collate function for text-based datasets like breast cancer."""
    *features, labels, masks = zip(*batch)
    # For text data, keep as list instead of stacking
    texts = list(features[0])  # features[0] contains the text strings
    labels = torch.tensor(labels)
    mask = masks[0]
    return (texts, labels, mask)


class CustomDataset(Dataset):
    def __init__(self, base_dataset, batch_size, num_clients, p_miss):
        self.dataset = base_dataset
        num_samples = len(base_dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(
            num_batches, num_clients, p_miss
        )
        self.batch_size = batch_size
        self.classes = base_dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch_idx = idx // self.batch_size
        data = self.dataset[idx]

        features, label = data[:-1], data[-1]

        mask = self.batch_patterns[batch_idx]
        return (*features, label, torch.tensor(mask, dtype=torch.bool))


class BreastCancerDataset(Dataset):
    def __init__(self, X, y, feature_names):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = y
        self.feature_names = feature_names
        self.classes = ["malignant", "benign"]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Return as tensor, not text
        return self.X[idx], self.y[idx]
