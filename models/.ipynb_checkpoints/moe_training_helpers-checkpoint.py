import torch
import numpy as np
from sklearn.metrics import f1_score
import sys

sys.path.append("./..")
from data.cifar_partitions import CIFAR_PARTITIONS
from data.data_utils import slice_cifar_block
from utils import from_mask_to_expert_with_full_info, tilde_powerset_except_empty

def count_wrong_routing_decisions(num_parties, masks, router_probs):
    """
    We will consider that a routing decision is wrong if the expert with maximum probability does not have full information
    """
    counter = 0
    alignments = tilde_powerset_except_empty(K=num_parties)
    for mask, probs in zip(masks, router_probs):
        correct_experts = from_mask_to_expert_with_full_info(
            mask=mask, tilde_power_set=alignments
        )
        top_expert = torch.argmax(probs)
        if top_expert not in correct_experts:
            counter += 1
    return counter

def get_block_from_input(x, i, num_parties):
    block_size = x.shape[1] // num_parties
    start_idx = i * block_size
    end_idx = (i + 1) * block_size
    return x[:, start_idx:end_idx]

def get_feature_names_for_client(feature_names, client_idx, num_clients):
    """Get feature names for a specific client block."""
    num_features_per_client = len(feature_names) // num_clients
    start_idx = client_idx * num_features_per_client
    end_idx = (client_idx + 1) * num_features_per_client
    return feature_names[start_idx:end_idx]
    
def row_to_text(row, col_names):
    parts = [f"{name}: {value:.4f}" for name, value in zip(col_names, row)]
    return " | ".join(parts)

def get_na_embedding(passive_encoder, passive_feature_names, device):
    """
    Precompute fixed N/A embedding for passive parties' missing data.
    """
    na_row = [float('nan')] * len(passive_feature_names)
    na_texts = [row_to_text(na_row, passive_feature_names)]
    
    with torch.no_grad():
        na_embedding = passive_encoder.encode(na_texts, convert_to_tensor=True).to(device)
    
    return na_embedding

def train_moe(participant_encoders, router_and_moe, dataloader_train, noisy_parties, optimizer, device):
    """
    Function to train the router and MoE based classification head.
    Args:
        - participant_encoders: frozen local encoders of each participants
        - router_and_moe: the model that comprises the router and MoPE layer
        - dataloader_train: dataloader with train data
        - noisy_parties: noisy/malicious participants (they must range from 0 to K-3)
        - optimizer: optimizer
        - device: training device
    """
    num_samples = len(dataloader_train.dataset)
    num_batches = len(dataloader_train)
    # Check total number of parties
    num_parties = len(participant_encoders)
    # Define variables to store information
    epoch_loss, epoch_acc = 0, 0
    entropy_loss, ce_loss = 0, 0 
    expert_routing_sum = torch.zeros(2 ** (num_parties - 1))
    epoch_masks = []
    epoch_routing_probs = []
    criterion = torch.nn.NLLLoss()
    # Set encoders on eval mode
    for autoencoder in participant_encoders:
        autoencoder.eval()
    # Iterate on the dataloader
    for batch_info in dataloader_train:
        party_representation_storage = {}
        images, labels, mask = batch_info
        epoch_masks.append(mask)
        images, labels, mask = images.to(device), labels.to(device), mask.to(device)
        # Get data representations of each participant
        with torch.no_grad():
            for party in range(num_parties):
                # print(f"Party {party}")
                # Check if we are on the active party, which is the last one and has access to every image patch
                if party != (num_parties - 1):
                    # If it is not the active party, check if the image patch is visible based on the mask
                    if not mask[party]:
                        # If the patch is not visible, simply take a vector of 0s as representation
                        # latent_dim = 1024
                        # latent_dim = 768
                        latent_dim = 384
                        # party_representation = torch.zeros(images.shape[0], latent_dim).to(device)
                        party_representation = torch.full((images.shape[0], latent_dim), -10.0).to(device)
                        party_representation_storage[party] = party_representation
                        continue
                # THIS IF STATEMENT CHOOSES NON NOISY PARTIES
                if party not in noisy_parties:
                    # print(f"Party {party} regular info")
                    party_patch_coordinates = CIFAR_PARTITIONS[num_parties][party]
                    party_data = slice_cifar_block(images, party_patch_coordinates)
                    party_representation = participant_encoders[party](party_data)
                    party_representation_storage[party] = party_representation
                # ELSE WE ADD NOISE
                elif mask[party]: # If the mask is true we add noise, otherwise we have 0s
                    # print(f"Party {party} added noise")
                    # latent_dim = 1024
                    # latent_dim = 768
                    latent_dim = 384
                    party_representation_storage[party] = (torch.randn(images.shape[0], latent_dim)*100).to(device)
        # Forward from MoE
        all_party_representations = []
        party = 0
        for party in range(num_parties):
            all_party_representations.append(party_representation_storage[party])
        all_party_representations = [rep.to(device) for rep in all_party_representations]
        # Concatenate along feature dimension
        combined_representation = torch.cat(all_party_representations, dim=1)
        # Forward from the router and MoE head
        optimizer.zero_grad()
        preds, probs = router_and_moe(
            combined_representation, party_representation_storage, training=False)
        # print(f"PROBS: {probs}")
        epoch_routing_probs.append(probs)
        # Compute accuracy
        epoch_acc += (
            torch.sum(torch.argmax(preds, dim=1) == labels).type(torch.float).item()
        )
        # Add routing probabilities per expert to compute the epoch average later
        expert_routing_sum += probs.sum(dim=0).detach().cpu()
        # print(f"ROUTING SUM: {expert_routing_sum}\n--------------\n")
        # Compute loss and update the router and MoE parameters
        loss_val = criterion(preds, labels)
        epoch_loss += loss_val.item()
        loss_val.backward()
        optimizer.step()

    # Compute mean epoch metrics
    epoch_loss /= num_batches
    epoch_acc /= num_samples
    print(f"FINAL EXPERT ROUTING SUM: {expert_routing_sum}")
    print(f"NUM SAMPLES: {num_samples}")
    expert_routing_sum /= num_samples
    

    return epoch_loss, epoch_acc, expert_routing_sum


def test_moe(participant_encoders, router_and_moe, dataloader_test, noisy_parties, device):
    """
    Function to test the router and MoE based classification head.
    Args:
        - participant_encoders: frozen local encoders of each participants
        - router_and_moe: the model that comprises the router and MoPE layer
        - dataloader_test: dataloader with test data
        - noisy_parties: noisy/malicious participants (they must range from 0 to K-3)
        - device: training device
    """
    num_samples = len(dataloader_test.dataset)
    num_batches = len(dataloader_test)
    # Check total number of parties
    num_parties = len(participant_encoders)
    # Define variables to store information
    epoch_loss, epoch_acc = 0, 0
    entropy_loss, ce_loss = 0, 0 
    expert_routing_sum = torch.zeros(2 ** (num_parties - 1))
    epoch_masks = []
    epoch_routing_probs = []
    criterion = torch.nn.NLLLoss()
    # Set encoders on eval mode
    for autoencoder in participant_encoders:
        autoencoder.eval()
    # Iterate on the dataloader
    for batch_info in dataloader_test:
        party_representation_storage = {}
        images, labels, mask = batch_info
        epoch_masks.append(mask)
        images, labels, mask = images.to(device), labels.to(device), mask.to(device)
        # Get data representations of each participant
        with torch.no_grad():
            for party in range(num_parties):
                # Check if we are on the active party, which is the last one and has access to every image patch
                if party != (num_parties - 1):
                    # If it is not the active party, check if the image patch is visible based on the mask
                    if not mask[party]:
                        # If the patch is not visible, simply take a vector of 0s as representation
                        # latent_dim = 1024
                        # latent_dim = 768
                        latent_dim = 384
                        # party_representation = torch.zeros(images.shape[0], latent_dim).to(device)
                        party_representation = torch.full((images.shape[0], latent_dim), -10.0).to(device)
                        party_representation_storage[party] = party_representation
                        continue
                # SELECT NON NOISY PARTY ON THE IF
                if party not in noisy_parties: # before != noisy_party for example if party != 0
                    party_patch_coordinates = CIFAR_PARTITIONS[num_parties][party]
                    party_data = slice_cifar_block(images, party_patch_coordinates)
                    party_representation = participant_encoders[party](party_data)
                    party_representation_storage[party] = party_representation
                elif mask[party]: # If the mask is true we add noise, otherwise we have 0s
                    # latent_dim = 1024
                    # latent_dim = 768
                    latent_dim = 384
                    party_representation_storage[party] = (torch.randn(images.shape[0], latent_dim)*100).to(device)
            # Forward from MoE
            all_party_representations = []
            for party in range(num_parties):
                all_party_representations.append(party_representation_storage[party])
            # Concatenate along feature dimension
            combined_representation = torch.cat(all_party_representations, dim=1)
            # Forward from the router and MoE head
            preds, probs = router_and_moe(
                combined_representation, party_representation_storage, training=False)
        epoch_routing_probs.append(probs)
        # Compute accuracy
        epoch_acc += (
            torch.sum(torch.argmax(preds, dim=1) == labels).type(torch.float).item()
        )
        # Add routing probabilities per expert to compute the epoch average later
        expert_routing_sum += probs.sum(dim=0).detach().cpu()
        # Compute loss and update the router and MoE parameters
        loss_val = criterion(preds, labels)
        epoch_loss += loss_val.item()
    # Compute mean epoch metrics
    epoch_loss /= num_batches
    epoch_acc /= num_samples
    expert_routing_sum /= num_samples
   

    return epoch_loss, epoch_acc, expert_routing_sum


def train_moe_tabular(participant_encoders, router_and_moe, dataloader_train, noisy_parties, optimizer, device):
    """
    Function to train the router and MoE based classification head for a single epoch on tabular data
    Args:
        - participant_encoders: frozen local encoders of each participants
        - router_and_moe: the model that comprises the router and MoPE layer
        - dataloader_train: dataloader with train data
        - noisy_parties: noisy/malicious participants (they must range from 0 to K-3)
        - optimizer: optimizer
        - device: training device
    """
    num_samples = len(dataloader_train.dataset)
    num_batches = len(dataloader_train)
    # Check total number of parties
    num_parties = len(participant_encoders)
    # Get feature names for the embeddings
    feature_names_all = dataloader_train.dataset.dataset.feature_names
    # Create N/A embeddings for padding
    passive_feature_names = get_feature_names_for_client(feature_names_all, 0, num_parties)
    na_embedding = get_na_embedding(participant_encoders[0], passive_feature_names, device)
    # Define variables to store information
    epoch_loss, epoch_acc = 0, 0
    entropy_loss, ce_loss = 0, 0 
    expert_routing_sum = torch.zeros(2 ** (num_parties - 1))
    epoch_masks = []
    epoch_routing_probs = []
    criterion = torch.nn.NLLLoss()
    # Storage for F1 computation
    all_predictions = []
    all_labels = []
    # Set encoders on eval mode
    for autoencoder in participant_encoders:
        autoencoder.eval()
    # Iterate on the dataloader
    for batch_info in dataloader_train:
        party_representation_storage = {}
        features, labels, mask = batch_info
        epoch_masks.append(mask)
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)
        # Get data representations of each participant
        with torch.no_grad():
            for party in range(num_parties):
                # print(f"Party {party}")
                # Check if we are on the active party, which is the last one and has access to every image patch
                if party != (num_parties - 1):
                    # If it is not the active party, check if the image patch is visible based on the mask
                    if not mask[party]:
                        # If the patch is not visible, simply take a vector of 0s as representation
                        latent_dim = 2560
                        # latent_dim = 1024
                        # latent_dim = 4096
                        party_representation = torch.full((features.shape[0], latent_dim), 0.0).to(device)
                        party_representation_storage[party] = party_representation
                        '''
                        party_representation = na_embedding.repeat(features.shape[0], 1)
                        party_representation_storage[party] = party_representation
                        '''
                        continue
                # THIS IF STATEMENT CHOOSES NON NOISY PARTIES
                if party not in noisy_parties:
                    party_data = get_block_from_input(features, party-len(noisy_parties), num_parties=2) # Subtract the number of noisy parties so that informative ones have access to data
                    feature_names = get_feature_names_for_client(feature_names_all, party, num_parties)
                    texts_party = [
                        row_to_text(party_data[i].detach().cpu().numpy(), feature_names) 
                        for i in range(len(party_data))
                    ]
                    party_representation = participant_encoders[party].encode(texts_party, convert_to_tensor=True)
                    party_representation_storage[party] = party_representation
                # ELSE WE ADD NOISE
                elif mask[party]: # If the mask is true we add noise, otherwise we have 0s
                    # print(f"Party {party} added noise")
                    latent_dim = 2560
                    # latent_dim = 1024
                    # latent_dim = 4096
                    party_representation_storage[party] = (torch.randn(features.shape[0], latent_dim)*100).to(device)
        # Forward from MoE
        all_party_representations = []
        party = 0
        for party in range(num_parties):
            all_party_representations.append(party_representation_storage[party])
        # Concatenate along feature dimension
        all_party_representations = [
            torch.tensor(rep) if isinstance(rep, np.ndarray) else rep 
            for rep in all_party_representations
        ]
        combined_representation = torch.cat(all_party_representations, dim=1)
        # Forward from the router and MoE head
        optimizer.zero_grad()
        preds, probs = router_and_moe(
            combined_representation, party_representation_storage, training=False
        )
        epoch_routing_probs.append(probs)
        # Get predictions and store for F1 computation
        batch_preds = torch.argmax(preds, dim=1)
        all_predictions.extend(batch_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Compute accuracy
        epoch_acc += (
            torch.sum(torch.argmax(preds, dim=1) == labels).type(torch.float).item()
        )
        # Add routing probabilities per expert to compute the epoch average later
        expert_routing_sum += probs.sum(dim=0).detach().cpu()
        # Compute loss and update the router and MoE parameters
        loss_val = criterion(preds, labels)
        epoch_loss += loss_val.item()
        loss_val.backward()
        optimizer.step()

    # Compute mean epoch metrics
    epoch_loss /= num_batches
    epoch_acc /= num_samples
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"FINAL EXPERT ROUTING SUM: {expert_routing_sum}")
    print(f"NUM SAMPLES: {num_samples}")
    expert_routing_sum /= num_samples
    

    return epoch_loss, epoch_acc, expert_routing_sum, epoch_f1


def test_moe_tabular(participant_encoders, router_and_moe, dataloader_test, noisy_parties, device):
    """
    Function to test the router and MoE based classification head for a single epoch on tabular data
    Args:
        - participant_encoders: frozen local encoders of each participants
        - router_and_moe: the model that comprises the router and MoPE layer
        - dataloader_test: dataloader with test data
        - noisy_parties: noisy/malicious participants (they must range from 0 to K-3)
        - device: training device
    """
    num_samples = len(dataloader_test.dataset)
    num_batches = len(dataloader_test)
    # Check total number of parties
    num_parties = len(participant_encoders)
    # Get feature names for the embeddings
    feature_names_all = dataloader_test.dataset.dataset.feature_names
    # Create N/A embeddings for alternative padding
    passive_feature_names = get_feature_names_for_client(feature_names_all, 0, num_parties)
    na_embedding = get_na_embedding(participant_encoders[0], passive_feature_names, device)
    # Define variables to store information
    epoch_loss, epoch_acc = 0, 0
    entropy_loss, ce_loss = 0, 0 
    expert_routing_sum = torch.zeros(2 ** (num_parties - 1))
    epoch_masks = []
    epoch_routing_probs = []
    criterion = torch.nn.NLLLoss()
    # Storage for F1 computation
    all_predictions = []
    all_labels = []
    # Set encoders on eval mode
    for autoencoder in participant_encoders:
        autoencoder.eval()
    # Iterate on the dataloader
    for batch_info in dataloader_test:
        party_representation_storage = {}
        features, labels, mask = batch_info
        epoch_masks.append(mask)
        features, labels, mask = features.to(device), labels.to(device), mask.to(device)
        # Get data representations of each participant
        with torch.no_grad():
            for party in range(num_parties):
                # Check if we are on the active party, which is the last one and has access to every image patch
                if party != (num_parties - 1):
                    # If it is not the active party, check if the image patch is visible based on the mask
                    if not mask[party]:
                        # If the patch is not visible, simply take a vector of 0s as representation
                        latent_dim = 2560
                        # latent_dim = 1024
                        # latent_dim = 4096
                        party_representation = torch.full((features.shape[0], latent_dim), 0.0).to(device)
                        party_representation_storage[party] = party_representation
                        '''
                        party_representation = na_embedding.repeat(features.shape[0], 1)
                        party_representation_storage[party] = party_representation
                        '''
                        continue
                # SELECT NON NOISY PARTY ON THE IF
                if party not in noisy_parties: # before != noisy_party for example if party != 0
                    party_data = get_block_from_input(features, party-len(noisy_parties), num_parties=2) # Subtract the number of noisy parties so that informative ones have access to data
                    feature_names = get_feature_names_for_client(feature_names_all, party, num_parties)
                    texts_party = [
                        row_to_text(party_data[i].detach().cpu().numpy(), feature_names) 
                        for i in range(len(party_data))
                    ]
                    party_representation = participant_encoders[party].encode(texts_party, convert_to_tensor=True)
                    party_representation_storage[party] = party_representation
                elif mask[party]: # If the mask is true we add noise, otherwise we have 0s
                    latent_dim = 2560
                    # latent_dim = 1024
                    # latent_dim = 4096
                    party_representation_storage[party] = (torch.randn(features.shape[0], latent_dim)*100).to(device)
            # Forward from MoE
            all_party_representations = []
            for party in range(num_parties):
                all_party_representations.append(party_representation_storage[party])
            # Concatenate along feature dimension
            all_party_representations = [
                torch.tensor(rep) if isinstance(rep, np.ndarray) else rep 
                for rep in all_party_representations
            ]
            combined_representation = torch.cat(all_party_representations, dim=1)
            # Forward from the router and MoE head
            preds, probs = router_and_moe(
                combined_representation, party_representation_storage, training=False
            )
        epoch_routing_probs.append(probs)
        # Get predictions and store for F1 computation
        batch_preds = torch.argmax(preds, dim=1)
        all_predictions.extend(batch_preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        # Compute accuracy
        epoch_acc += (
            torch.sum(torch.argmax(preds, dim=1) == labels).type(torch.float).item()
        )
        # Add routing probabilities per expert to compute the epoch average later
        expert_routing_sum += probs.sum(dim=0).detach().cpu()
        # Compute loss and update the router and MoE parameters
        loss_val = criterion(preds, labels)
        epoch_loss += loss_val.item()
    # Compute mean epoch metrics
    epoch_loss /= num_batches
    epoch_acc /= num_samples
    expert_routing_sum /= num_samples
    epoch_f1 = f1_score(all_labels, all_predictions, average='macro')


    return epoch_loss, epoch_acc, expert_routing_sum, epoch_f1