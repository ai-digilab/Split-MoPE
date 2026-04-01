import torch
import time

def get_dino_encoder(model_ref='dinov2_vits14'):
    '''
    Func that downloads a pretrained dinoV2 encoder
    '''
    dino_vit = torch.hub.load('facebookresearch/dinov2', model_ref)
    return dino_vit