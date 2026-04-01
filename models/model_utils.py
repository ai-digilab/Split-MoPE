import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

sys.path.append("./..")

from data.cifar_partitions import CIFAR_PARTITIONS
from utils import tilde_powerset_except_empty


def task_to_dims(dataset, num_parties, feature_extractors):
    """
    Function that takes dataset and number of parties and local feature extractor as input;
    Computes the needed router input dimensionality, the one for each classifier and output dims
    """
    router_input_dim, classifier_output_dim, expert_input_dims = (
        None,
        None,
        None,
    )

    if dataset == "cifar10":
        # CIFAR-10 has 10 classes
        classifier_output_dim = 10
        # Compute the output dimension of each participant and router input dimension
        encoder_output_dims, router_input_dim = generate_dummy_images_check_output(
            num_parties=num_parties, encoders=feature_extractors
        )
    elif dataset == "cifar100":
        # CIFAR-100 has 100 classes
        classifier_output_dim = 100
        # Compute the output dimension of each participant and router input dimension
        encoder_output_dims, router_input_dim = generate_dummy_images_check_output(
            num_parties=num_parties, encoders=feature_extractors
        )
    elif dataset == "breast_cancer":
        classifier_output_dim = 2
        encoder_output_dims = {party: 2560 for party in range(num_parties)}
        router_input_dim = 2560 * num_parties
    else:
        raise ValueError(f"Unavailable dataset {dataset}")

    return router_input_dim, classifier_output_dim, encoder_output_dims


def generate_dummy_images_check_output(num_parties, encoders):
    router_input_dim = 0
    participants = CIFAR_PARTITIONS[num_parties]
    device = next(encoders[0].parameters()).device
    # Generate dummy inputs based on pixel ranges
    encoder_output_dims = {}

    for party, (height_range, width_range) in participants.items():
        # Calculate image dimensions
        height = height_range[1] - height_range[0]
        width = width_range[1] - width_range[0]
        channels = 3  # RGB channels
        if height != 0 and width != 0:
            # Create random tensor with image shape [channels, height, width]
            dummy_input = torch.rand(1, channels, height, width).to(device)
    
            # Make a forward pass using the encoder
            party_encoder = encoders[party]
            output = party_encoder(dummy_input)
            encoder_output_dims[party] = output.shape[1]
            router_input_dim += output.shape[1]
        else:
            encoder_output_dims[party] = 384
            router_input_dim += 384
    return encoder_output_dims, router_input_dim


def compute_expert_input_dims(num_parties, encoder_output_dims):
    """
    Compute the input size of each expert, taking into account the output dimension of each participant's encoder
    """
    expert_input_dims = []
    # Generate all possible interesting alignments
    alignments = tilde_powerset_except_empty(num_parties)
    # Iterate on alignments and check the dimensionality of each participant's output;
    # Add dimensions participants of alignment to compute total input dim
    for alignment in alignments:
        expert_input_dim = 0
        for party in alignment:  # Iterate over parties in this alignment
            if isinstance(encoder_output_dims, int):
                expert_input_dim += encoder_output_dims
            else:
                expert_input_dim += encoder_output_dims[party]
        expert_input_dims.append(expert_input_dim)
    return expert_input_dims