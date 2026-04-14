import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from flwr.app import Context, RecordDict, Message
from flwr.serverapp import Grid, ServerApp
from flwr.common import Metadata, ArrayRecord, MetricRecord, ConfigRecord
from pytorchexample.task import ServerHead

from sklearn.metrics import f1_score

import time

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Vertical Federated Learning Orchestrator for Tabular Data.
    """
    # 1. Get the list of all available client nodes
    node_ids = list(grid.get_node_ids())
    print(f"Found {len(node_ids)} nodes: {node_ids}")

    print("\n[STEP 1] Requesting training embeddings from all clients...")
    
    # We send a 'train' message to every node to get their local features
    train_messages = []
    for i, node_id in enumerate(node_ids):
        apply_mask = (i != len(node_ids) - 1)
        content = RecordDict()
        content.config_records["node_params"] = ConfigRecord({"apply_mask": apply_mask})

        msg = Message(
            content=content,
            metadata=Metadata(
                run_id=context.run_id,
                message_id="",      
                src_node_id=len(node_ids), 
                dst_node_id=node_id,
                reply_to_message_id="",
                group_id="vfl_train",
                ttl=3600.0,
                message_type="train",
                created_at=time.time()
            ),
        )
        train_messages.append(msg)

    client_train_embeddings = []
    train_labels = None

    # Dispatch all messages and wait for the replies
    train_replies = list(grid.send_and_receive(train_messages))

    for reply in train_replies:
        # Check if the message actually has content
        if reply.has_content():
            emb_np = reply.content.arrays.to_numpy_ndarrays()[0]
            client_train_embeddings.append(torch.from_numpy(emb_np))
            
            if train_labels is None:
                train_labels = torch.tensor(reply.content.metrics["labels"], dtype=torch.long)
        else:
            print(f"Node {reply.metadata.src_node_id} returned an empty message!") 

    # Concatenate features from all clients side-by-side
    combined_train_features = torch.cat(client_train_embeddings, dim=1)
    print(f"-> Combined Feature Vector Shape: {combined_train_features.shape}")

    print("\n[STEP 2] Training the Server Head on combined features...")
    # Create the unified dataset on the server
    train_ds = TensorDataset(combined_train_features, train_labels)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Initialize the Head. 
    num_clients = len(train_replies)
    head_model = ServerHead(num_clients=num_clients, client_embedding_dim=32, num_classes=2)
    
    optimizer = torch.optim.Adam(head_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = context.run_config["num-server-rounds"]

    for epoch in range(num_epochs):
        head_model.train()
        epoch_loss = 0.0
        for batch_feat, batch_label in train_loader:
            optimizer.zero_grad()
            outputs = head_model(batch_feat)
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {epoch_loss/len(train_loader):.4f}")

    print("\n[STEP 3] Requesting test embeddings...")

    eval_messages = []
    for i, node_id in enumerate(node_ids):
        apply_mask = (i != len(node_ids) - 1)
        
        content = RecordDict()
        content.config_records["node_params"] = ConfigRecord({"apply_mask": apply_mask})

        msg = Message(
            content=content,
            metadata=Metadata(
                run_id=context.run_id,
                message_id="",
                src_node_id=len(node_ids),
                dst_node_id=node_id,
                reply_to_message_id="",
                group_id="vfl_eval",
                ttl=3600.0,
                message_type="evaluate",
                created_at=time.time()
            ),
        )
        eval_messages.append(msg)

    eval_replies = list(grid.send_and_receive(eval_messages))

    client_test_embeddings = []
    test_labels = None

    for reply in eval_replies:
        if reply.has_content():
            test_emb_np = reply.content.arrays.to_numpy_ndarrays()[0]
            client_test_embeddings.append(torch.from_numpy(test_emb_np))
            
            if test_labels is None:
                test_labels = torch.tensor(reply.content.metrics["test_labels"], dtype=torch.long)

    combined_test_features = torch.cat(client_test_embeddings, dim=1)
    test_loader = DataLoader(TensorDataset(combined_test_features, test_labels), batch_size=32)

    # Evaluation loop
    head_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for feats, targets in test_loader:
            outputs = head_model(feats)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(targets, predicted)
    print(f"\nFinal VFL Test Accuracy on Sklearn Dataset: {accuracy:.2f}%\n Final VFL Test F1: {f1: .2f}")