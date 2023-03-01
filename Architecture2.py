import torch
import torch.nn as nn
from torchvision import models


DEBUG = True
def printf(*args):
    if DEBUG:
        print(" ".join(map(str, args)))
    else:
        pass
    
    


class Architecture2(nn.Module):
    def __init__(self,  hidden_size,embedding_size , no_layers, vocab_size, max_lengths = 50):
        super().__init__()
        
        printf("Not implemented yet")
        # TODO
    
    def forward(self, img, cap):
        # TODO
        printf("Not implemented yet")
        
             
    def caption_images(self, img, vocab_dict, max_length = 20, is_deterministic = False, temp = 0.1): ## FOR TESTING
        # TODO
        printf("Not implemented yet")
    
