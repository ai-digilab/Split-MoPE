import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Message, RecordDict, ArrayRecord, MetricRecord, Metadata

from pytorchexample.task import ClientBackbone, load_sklearn_data

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    node_params = msg.content.config_records.get("node_params", {})    # 1. Setup Partition Info
    print(f"NODE PARAMS {node_params}")
    apply_mask = node_params.get("apply_mask", True)
    print(f"APPLY MASK {apply_mask}")
    partition_id = context.node_config["partition-id"]
    print(f"PARTITION ID {partition_id}")
    num_partitions = context.node_config["num-partitions"]
    p_miss_train = context.run_config.get("p-miss-train", 0.0001)
    print(f"P MISS TRAIN: {p_miss_train}")
    batch_size = context.run_config.get("batch-size", 32)
    print(f"BATCH SIZE: {batch_size}")
    # 2. Load the Sklearn Data 
    trainloader, _ = load_sklearn_data(partition_id, num_partitions, p_miss_train=p_miss_train, batch_size=batch_size)

    # 3. Dynamic Model Initialization
    # We peek at the first batch to see how many features this client has
    first_batch = next(iter(trainloader))
    features_tensor = first_batch[0] 
    input_dim = features_tensor.shape[1] 
    batch_size = features_tensor.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the MLP backbone with the correct input dimension
    model = ClientBackbone(input_dim=input_dim).to(device)
    model.eval()
    # Dynamic embedding shape obtention
    embeddings = model(features_tensor)
    embedding_size = embeddings.shape[1]

    all_embeddings = []
    all_labels = []

    # 4. Forward Pass only
    with torch.no_grad():
        for batch in trainloader:
            x, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if apply_mask:
                print(f"APPLY MASK: {apply_mask}")
                batch_is_visible = mask[0].item()
                if batch_is_visible:
                    embeddings = model(x)
                else:
                    embeddings = torch.zeros(x.shape[0], embedding_size).to(device)
            else:
                embeddings = model(x) 

            all_embeddings.append(embeddings.cpu())
            all_labels.append(y.cpu())

    # 5. Package for Server
    full_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    full_labels = torch.cat(all_labels, dim=0).numpy().tolist()

    if isinstance(full_embeddings, torch.Tensor):
        embeddings_np = full_embeddings.detach().cpu().numpy()
    else:
        embeddings_np = full_embeddings

    # Pass it
    res_arrays = ArrayRecord(numpy_ndarrays=[embeddings_np])

    content = RecordDict()
    content.arrays = res_arrays
    content.metrics = MetricRecord({"labels": full_labels})

    return msg.create_reply(content=content)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    # 1. Read node params (same as train)
    node_params = msg.content.config_records.get("node_params", {})
    apply_mask = node_params.get("apply_mask", True)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    p_miss_test = context.run_config.get("p-miss-test", 0.0001)
    print(f"P MISS TEST: {p_miss_test}")
    batch_size = context.run_config.get("batch-size", 32)
    print(f"BATCH SIZE: {batch_size}")

    # 2. Load data
    _, testloader = load_sklearn_data(partition_id, num_partitions, p_miss_test=p_miss_test, batch_size=batch_size)

    # 3. Setup model
    first_batch = next(iter(testloader))
    input_dim = first_batch[0].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClientBackbone(input_dim=input_dim).to(device)
    model.eval()

    # Get embedding size from a dry run
    embedding_size = model(first_batch[0].to(device)).shape[1]

    all_test_embeddings = []
    all_test_labels = []

    # 4. Generate embeddings with mask support
    with torch.no_grad():
        for batch in testloader:
            x, y, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if apply_mask:
                batch_is_visible = mask[0].item()
                if batch_is_visible:
                    embeddings = model(x)
                else:
                    embeddings = torch.zeros(x.shape[0], embedding_size).to(device)
            else:
                embeddings = model(x)

            all_test_embeddings.append(embeddings.cpu())
            all_test_labels.append(y.cpu())

    full_test_embeddings = torch.cat(all_test_embeddings, dim=0).numpy()
    full_test_labels = torch.cat(all_test_labels, dim=0).numpy().tolist()

    # 5. Package and reply
    content = RecordDict()
    content.arrays = ArrayRecord(numpy_ndarrays=[full_test_embeddings])
    content.metrics = MetricRecord({"test_labels": full_test_labels})

    return msg.create_reply(content=content)