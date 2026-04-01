import random
from itertools import chain, combinations

import numpy as np
import torch
import yaml

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def tilde_powerset_except_empty(K):
    """
    Function to compute the set of interesting alignments, i.e. the ones where the active participant is;
    We assume that the active is the Kth participant.
    """
    powerset = list(
        chain.from_iterable(combinations(range(K), r) for r in range(1, K + 1))
    )
    tilde_powerset = [subset for subset in powerset if K - 1 in subset]
    return tilde_powerset

def from_mask_to_expert_with_full_info(mask, tilde_power_set):
    """
    Knowing the mask and all the possible alignmnets, we check which experts have full information, i.e. no 0s as input
    """
    # Create set of parties with information (including the active participant)
    parties_with_information = {i for i, value in enumerate(mask) if value}
    parties_with_information.add(len(mask))

    # Use list comprehension to find alignments where all parties have information
    return [
        idx
        for idx, alignment in enumerate(tilde_power_set)
        if parties_with_information.issuperset(alignment)
    ]

def check_visible_data_fraction_per_participant(dataloader, n_participants):
    '''
    Having the mask of the dataloader, check how much data does each party have access to
    '''
    all_epoch_masks = []
    samples = len(dataloader.dataset)
    total_visible = None
    for batch_info in dataloader:
        x, _, mask = batch_info
        batch_size = x.shape[0]
        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
        if total_visible is None:
            total_visible = np.zeros(n_participants - 1, dtype=np.int64) # Subtract one since the active participant sees all the blocks
            
        total_visible += mask_np * batch_size  # sum over batch samples for each participant, take into account that there is a single mask per batch
    # Divide by total number of samples to get fractions per participant
    fractions = total_visible / samples
    return fractions
