import torch
import time

def get_dino_encoder(model_ref='dinov2_vits14'):
    '''
    Func that downloads a pretrained dinoV2 encoder
    '''
    dino_vit = torch.hub.load('facebookresearch/dinov2', model_ref)
    return dino_vit

def generate_dummy_input_and_forward(model_ref='dinov2_vits14'):
    # Load the model
    dino_vit = get_dino_encoder(model_ref)
    dino_vit = dino_vit.to("cuda")
    print("Aaaaaaaaaaaaaaaaa")
    time.sleep(60)
    dino_vit.eval()
    
    # Create dummy input tensor with batch size 1, 3 channels, 224x224 image size
    dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
    
    # Forward pass
    with torch.no_grad():
        output = dino_vit(dummy_input)
    
    return output, dummy_input

if __name__ == "__main__":
    # Get model and perform forward pass
    output, dummy_input = generate_dummy_input_and_forward()
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")