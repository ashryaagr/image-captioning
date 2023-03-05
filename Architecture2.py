import torch
import torch.nn as nn
from torchvision import models


DEBUG = True
def printf(*args):
    if DEBUG:
        print(" ".join(map(str, args)))
    else:
        pass

class Encoder(nn.Module):
    def __init__(self, encoding_size):
        super(Encoder, self).__init__()
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)

        self.linear = nn.Linear(model.fc.in_features, encoding_size)
        self.fine_tune()

    def forward(self, images):
        y = self.model(images)
        y = y.view(y.size(0),-1)
        y = self.linear(y)
        return y

    def fine_tune(self, fine_tune=False):
        for p in self.model.parameters():
            p.requires_grad = False
        # Choose which layers to fine-tune below. Perhaps can think if time remains before assignment deadline
        for c in list(self.model.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed_dim, self.hidden_dim, self.vocab_size, self.num_layers = embed_dim, hidden_dim, vocab_size, num_layers
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size = 2* embed_dim, hidden_size = hidden_dim, num_layers = num_layers, batch_first=True)
        self.linearDec = nn.Linear(self.hidden_dim, self.vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, y, cap):
        cap = torch.cat((torch.zeros(cap.shape[0], 1, dtype=int).to(self.device), cap), dim=1)
        embed = self.embed(cap)
        embed = torch.cat((y.unsqueeze(1).repeat(1, embed.shape[1], 1),embed), dim =2 )# [64, 18, 300] [64, 1, 300]
        hiddens , _ = self.lstm(embed)
        outputs = self.linearDec(hiddens)
        return outputs

class Architecture2(nn.Module):
    def __init__(self,  hidden_size,embedding_size , no_layers, vocab_size, max_lengths = 50, applySoftmax=False):
        super(Architecture2, self).__init__()
        
        self.embedding_size = embedding_size
        self.embedding_size = hidden_size
        self.vocab_size = vocab_size
        self.no_layers = no_layers

        # Turn this flag on if we use NLL Loss. Keep false for cross entropy
        self.applySoftmax = applySoftmax
        
        self.encoder = Encoder(self.embedding_size)
        self.decoder = Decoder(self.embedding_size, self.embedding_size, self.vocab_size, self.no_layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, img, cap):
        
        y = self.encoder(img)
        outputs = self.decoder(y, cap)

        if self.applySoftmax:
            outputs = self.softmax(outputs)
        
        return outputs
             
    def caption_images(self, img, vocab_dict, max_length = 20, is_deterministic = False, temp = 0.1):
        # For generating captions and at test time
        caption = []
        
        with torch.no_grad():
            img_features = self.encoder(img)
            img_features = img_features.unsqueeze(1)

            # get embedding for the padding token. <pad> token index is 0
            tok = torch.zeros((img_features.shape[0]), dtype=int).to(self.device)

            states = None

            for i in range(max_length):
                y = self.decoder.embed(tok)
                y = y.unsqueeze(1)
                hiddens , states = self.decoder.lstm(torch.cat((img_features, y), dim=-1), states)
                output = self.decoder.linearDec(hiddens.squeeze(1))
            
                if is_deterministic: 
                    tok = output.argmax(1)
                    
                else:
                    tok = self.softmax(output/temp)
                    tok = torch.multinomial(tok,1).view(1)
                
                caption.append(vocab_dict[tok.item()])
                
                if caption[-1] == "<EOS>":
                    break

        return caption
    
