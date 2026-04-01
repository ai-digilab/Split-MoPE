from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name="Qwen/Qwen3-Embedding-4B"):
    model = SentenceTransformer(model_name)
    # Freeze params
    for param in model.parameters():
        param.requires_grad = False
    return model