from data.data_utils import get_dataloaders
from models.moe import MoE_with_router_sigmoid
from models.dino_encoders import get_dino_encoder
from models.moe_training_helpers import train_moe, test_moe
from utils import set_seed

import torch
import wandb

# Define configuration
config = {"dataset": "cifar10", "batch_size": 256, "num_workers": 0}

for run in [0]:
    for p_miss in [0.999]:
        class Args:
            def __init__(self):
                self.num_clients = 2
                self.entropy_weight = 0
                self.training_epochs = 60
                self.p_miss_train = p_miss
                self.p_miss_test = p_miss
                self.seed = 42 + run
        
        
        args = Args()
        
        entropy_weight = 0
        # Inint wandb
        wandb.init(
            project="MoPE_run",
            name=f"{args.num_clients}_participants_p_miss_train_test_{args.p_miss_train}_seed_{args.seed}",
            reinit = True
        )
        # Set seed
        set_seed(args.seed)
        # Set device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Get dataloader
        train_loader, test_loader = get_dataloaders(args, config, p_miss_test=args.p_miss_test)
        # Instantiate local encoders for participants   
        local_encoder = get_dino_encoder() # dinov2_vitb14 by default
        local_encoder = local_encoder.to(device)
        participant_encoders = [local_encoder, local_encoder]
        # Ensure that params are frozen - non trainable local encoders
        for param in local_encoder.parameters():
            param.requires_grad = False
        # Create the MoPE
        MoE = MoE_with_router_sigmoid(
            num_parties=args.num_clients,
            dataset=config["dataset"],
            encoders=participant_encoders,
        ).to(device)
        # Instantiate optimizer
        optimizer = torch.optim.Adam(MoE.parameters(), lr = 0.0001)
        # Training loop
        for epoch in range(args.training_epochs):
            # Train for an epoch
            (   epoch_loss_train,
                epoch_acc_train,
                routing_probs_train,
            ) = train_moe(
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
                }
            )
            # Testing loop
            (
                epoch_loss_test,
                epoch_acc_test,
                routing_probs_test,
            ) = test_moe(
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
                }
            )
            torch.cuda.empty_cache()
        wandb.finish()
