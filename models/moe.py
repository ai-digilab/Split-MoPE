import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .model_utils import task_to_dims, compute_expert_input_dims

sys.path.append("./..")
from utils import tilde_powerset_except_empty

class MoE_with_router_sigmoid(nn.Module):
    def __init__(self, num_parties, dataset, encoders):
        super(MoE_with_router_sigmoid, self).__init__()
        self.num_parties = num_parties
        self.dataset = dataset
        self.encoders = encoders
        self.alignments_per_experts = tilde_powerset_except_empty(self.num_parties)
        print(f"\n\nAlignments per expert:{self.alignments_per_experts}\n\n")
        # Compute input output dimensions for the components
        (
            self.router_input_dim,
            self.encoder_output_dims,
            self.expert_input_dims,
            self.classifier_output_dim,
        ) = self.obtain_model_dimensions(
            dataset=self.dataset,
            num_parties=self.num_parties,
            feature_extractors=self.encoders,
        )
        print(self.router_input_dim)
        # Instantiate a router
        self.router = router(
            input_dim=self.router_input_dim, output_dim=len(self.alignments_per_experts)
        )
        # Create the experts
        self.experts = nn.ModuleList()
        print(f"Expert input dims: {self.expert_input_dims}\n")
        for expert_input_dim in self.expert_input_dims:
            expert = expert_net(
                input_dim=expert_input_dim, output_dim=self.classifier_output_dim
            )
            self.experts.append(expert)

    def obtain_model_dimensions(self, dataset, num_parties, feature_extractors):
        router_input_dim, classifier_output_dim, encoder_output_dims = task_to_dims(
            dataset, num_parties, feature_extractors
        )
        print(encoder_output_dims)
        expert_input_dims = compute_expert_input_dims(
            num_parties=num_parties, encoder_output_dims=encoder_output_dims
        )

        return (
            router_input_dim,
            encoder_output_dims,
            expert_input_dims,
            classifier_output_dim,
        )

    def forward(self, x, party_embeddings, training):
        """
        x: concatenated embeddings
        party_embeddings: dict that defines the embeddings corresponding to each party
        """
        expert_input_data = {}
        # Get gating weights from router (sigmoid outputs in [0, 1])
        router_gates = self.router(x, add_noise=training)  
        # Prepare input data for each expert according to alignments
        for expert_idx, alignment in enumerate(self.alignments_per_experts):
            current_alignment_data = [party_embeddings[party] for party in alignment]
            expert_input_data[expert_idx] = torch.cat(current_alignment_data, dim=1)
    
        expert_softmax_outputs = []
    
        # Get softmax output from each expert and weight by gating
        for expert_idx, expert in enumerate(self.experts):
            data = expert_input_data[expert_idx]
            batch_gates = router_gates[:, expert_idx].view(-1, 1)  
            expert_logits = expert(data)  
            expert_softmax = F.softmax(expert_logits, dim=1) 
            weighted_softmax = batch_gates * expert_softmax  
            expert_softmax_outputs.append(weighted_softmax)
    
        # Sum weighted expert outputs
        ensemble_probs = torch.stack(expert_softmax_outputs, dim=0).sum(dim=0)  
        # Renormalize across classes to ensure output sums to 1
        ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)
        # Add numerical stability, take log for NLLLoss
        epsilon = 1e-8
        log_probs = torch.log(torch.clamp(ensemble_probs, epsilon, 1.0))
    
        # Return log-probs with correct normalization, and the router gating weights
        return log_probs, router_gates

class router(nn.Module):
    def __init__(self, input_dim, output_dim, noise_std=0.1, epsilon=0.3):
        super(router, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Define the layers
        self.fc_1 = nn.Linear(
            in_features=self.input_dim, out_features=int(self.input_dim * 2)
        )
        self.fc_2 = nn.Linear(
            in_features=int(self.input_dim * 2), out_features=self.output_dim
        )
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.noise_std = noise_std
        self.epsilon = epsilon
    def forward(self, x, add_noise=False):
        x = self.relu(self.fc_1(x))
        x = self.sigmoid(self.fc_2(x))
        return x

class expert_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(expert_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Define the layers
        self.fc_1 = nn.Linear(
            in_features=self.input_dim, out_features=int(self.input_dim * 2)
        )
        self.fc_2 = nn.Linear(
            in_features=int(self.input_dim * 2), out_features=self.output_dim
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.fc_1(x))
        x = self.fc_2(x)
        
        return x

