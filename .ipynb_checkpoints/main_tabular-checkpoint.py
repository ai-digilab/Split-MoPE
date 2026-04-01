from data.data_utils import get_dataloaders
from models.moe import MoE_with_router_sigmoid
from models.embedding_model import get_embedding_model
from models.moe_training_helpers import train_moe_tabular, test_moe_tabular
from utils import set_seed

import torch
import wandb
import gc

# Define configuration
config = {"dataset": "breast_cancer", "batch_size": 25, "num_workers": 0}

for run in [0, 1, 2]:
    for p_miss in [0.0001, 0.1, 0.5, 0.6]:
        class Args:
            def __init__(self):
                self.num_clients = 2
                self.entropy_weight = 0
                self.training_epochs = 300
                self.p_miss_train = p_miss
                self.p_miss_test = p_miss
                self.seed = 42 + run
        
        
        args = Args()
        
        entropy_weight = 0
        # Inint wandb
        wandb.init(
            project="MoPE_run_tabular",
            name=f"P_miss_train_test_{p_miss}_seed_{args.seed}",
            reinit = True
        )
        # Set seed
        set_seed(args.seed)
        # Set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(device)
        # Get dataloader
        train_loader, test_loader = get_dataloaders(args, config, p_miss_test=args.p_miss_test)
        # Instantiate local encoders for participants; autoencoder architecture for encoder training    
        local_encoder = get_embedding_model() # "Qwen/Qwen3-Embedding-0.6B"
        local_encoder = local_encoder.to(device)
        participant_encoders = [local_encoder, local_encoder]
        # Ensure that params are frozen
        for param in local_encoder.parameters():
            param.requires_grad = False
            
        MoE = MoE_with_router_sigmoid(
            num_parties=args.num_clients,
            dataset=config["dataset"],
            encoders=participant_encoders,
        ).to(device)
        optimizer = torch.optim.Adam(MoE.parameters(), lr = 0.0001)
        # Training loop
        for epoch in range(args.training_epochs):
            # Train for an epoch
            (   epoch_loss_train,
                epoch_acc_train,
                routing_probs_train,
                f1
            ) = train_moe_tabular(
                participant_encoders=participant_encoders,
                router_and_moe=MoE,
                dataloader_train=train_loader,
                noisy_parties = [],
                optimizer=optimizer,
                device=device,
            )
            # Log training stats to Wandb
            for i, prob in enumerate(routing_probs_train):
                wandb.log({f"routing_sigmoid_weights_train/expert_{i}": prob.item(), "epoch": epoch})
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss_total": epoch_loss_train,
                    "train/acc": epoch_acc_train,
                    "train/f1": f1
                }
            )
            (
                epoch_loss_test,
                epoch_acc_test,
                routing_probs_test,
                f1
            ) = test_moe_tabular(
                participant_encoders=participant_encoders,
                router_and_moe=MoE,
                dataloader_test=test_loader,
                noisy_parties=[],
                device=device,
            )
            # Log testing stats to Wandb
            for i, prob in enumerate(routing_probs_test):
                wandb.log({f"routing_sigmoid_weights_test/expert_{i}": prob.item(), "epoch": epoch})
            wandb.log(
                {
                    "epoch": epoch,
                    "test/loss": epoch_loss_test,
                    "test/acc": epoch_acc_test,
                    "test/f1": f1
                }
            )
            torch.cuda.empty_cache()
        wandb.finish()
        
        if torch.cuda.is_available():
            optimizer.zero_grad(set_to_none=True)
    
        # Delete models, optimizers, and references
        del optimizer, MoE
        del participant_encoders
    
        # Move encoder to CPU and delete
        local_encoder.cpu()
        del local_encoder
    
        # Clear dataloaders
        del train_loader, test_loader
    
        # Force cleanup
        gc.collect()
    
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(
                f"After p_miss={p_miss} cleanup - GPU memory allocated: "
                f"{torch.cuda.memory_allocated(device) / 1e9:.2f} GB"
            )
            print(
                f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB\n\n"
            )