import torch
import torch.nn as nn
from torchvision import models


DEBUG = False
def printf(*args):
    if DEBUG:
        print(" ".join(map(str, args)))
    else:
        pass


class ResNetRNN(nn.Module):
    def __init__(self,  hidden_size,embedding_size , no_layers, vocab_size, max_lengths = 50):
        super().__init__()
        
        ## COMMON SHIZ
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        
        self.relu = nn.ReLU(inplace=False) 
        self.bn = nn.BatchNorm1d(self.embedding_size, momentum=0.01)
        
        ## ENCODER LAYERS
        self.model = models.resnet50(pretrained=True)  ## Imported ResNet50
        
        self.fc_in_feature = self.model.fc.in_features  ## To get the output feature vector dimension of the penultimate layer
        
        self.model = nn.Sequential(*list(self.model.children())[:-1]) ## Removing last layer 
        
        for param in self.model.parameters():
            param.requires_grad = False ## Disabling backpop in Resnet layers
           
        self.linear = nn.Linear(self.fc_in_feature, self.embedding_size) # torch dim were [64,2048,1,1] so multiplied to get the number of in features
        
        self.vocab_size = vocab_size
        self.no_layers = no_layers
                
               
        ## DECODER LAYERS
        self.embed = nn.Embedding(self.vocab_size, self.embedding_size)
        # self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.hidden_size, num_layers = self.no_layers, batch_first=True) ##LSTM BLOCK

        self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.no_layers, batch_first=True) ##RNN BLOCK

        self.linearDec = nn.Linear(self.hidden_size, self.vocab_size)
    
    def forward(self, img, cap):
        ## ENCODER FORWARD
        y = self.model(img) #RESNET50
        printf("feat after resnet", y.shape)
        
        y = y.view(y.size(0),-1) # get the 1D vector size for the linear layer
        printf("feat after reshape to 1 d", y.shape)
        
#         y = self.bn(self.linear(y)) # Linear Layer + batch norm
        y = self.linear(y)
#         printf("feat after linear", y.shape)
        
        ## DECODER FORWARD
        printf("caption before",cap.shape) # [6, 13]
        embed = self.embed(cap)
        printf("embed ",embed.shape) # [6, 13, 300]
        printf("vocab size", self.vocab_size) 
        
        printf("y before unsqueezing", y.shape) # [6, 300]
        
        y = y.unsqueeze(1)# [300] -> [6, 1, 300]
        # caption: [6, 13, 300]

        embed = torch.cat((y,embed), dim =1 ) 

        hiddens , _ = self.rnn(embed)
        
        printf("hiddens", hiddens.shape)
        
        outputs = self.linearDec(hiddens)
        printf("outputs", outputs.shape)
        
        return outputs
             
    def caption_images(self, img, vocab_dict, max_length = 20, is_deterministic = False, temp = 0.1): ## FOR TESTING
        result = []
        
        states = None
        
        with torch.no_grad():
            y = self.model(img) #RESNET50
            y = y.view(y.size(0),-1) # get the 1D vector size for the linear layer
            y = self.linear(y) # Linear Layer + batch norm
            
            y = y.unsqueeze(1)   
            for i in range(max_length): 
#                 printf("im inside")
                hiddens , states = self.rnn(y,states)
                hiddens = hiddens.squeeze(1)
        
                output = self.linearDec(hiddens)
            
                if is_deterministic: 
                    pred = output.argmax(1)
                    
                else:
                    ## Stochastic sampling for distribution 
                    pred = nn.functional.softmax(output/temp, dim = 1 )
                    pred = torch.multinomial(pred,1).view(1) #sampling without replacement
                  
                    
                result.append(pred.item())
                y = self.embed(pred).unsqueeze(1)

                if vocab_dict[pred.item()] == "<EOS>":
                    break

        return [vocab_dict[i] for i in result]
    
