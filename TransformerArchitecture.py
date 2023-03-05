import torch
import torch.nn as nn
from torchvision import models


DEBUG = False
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


# class Attention(nn.Module):
#     def __init__(self, encoder_dim, decoder_dim, attention_dim):
#         super(Attention, self).__init__()
#         self.encoder_att = nn.Linear(encoder_dim, attention_dim)
#         self.decoder_att = nn.Linear(decoder_dim, attention_dim)
#         self.full_att = nn.Linear(attention_dim, 1)
#         self.relu = nn.ReLU()
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, encoder_out, decoder_hidden):
#         att1 = self.encoder_att(encoder_out)
#         att2 = self.decoder_att(decoder_hidden)
#         att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
#         alpha = self.softmax(att)
#         attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
#         return attention_weighted_encoding, alpha

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embed_dim, self.hidden_dim, self.vocab_size, self.num_layers = embed_dim, hidden_dim, vocab_size, num_layers
        self.embed = nn.Embedding(self.vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size = 2* embed_dim, hidden_size = 2*hidden_dim, num_layers = num_layers, batch_first=True)
        
        # Transformer layers
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_dim*2,
                                                                    nhead=8,
                                                                    dim_feedforward=self.hidden_dim*4,
                                                                    dropout=0.1,
                                                                    activation='relu')
        
        # creating a stack of transformer decoder layers (we can change the number of layers)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer,
                                                         num_layers=3)
        
        print("--------------Using Transformer Architecture --------------!!")
        
        self.linearDec = nn.Linear(2*self.hidden_dim, self.vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, y, cap):
        cap = torch.cat((torch.zeros(cap.shape[0], 1, dtype=int).to(self.device), cap), dim=1)#shape: (batch_size, max_length, embed_dim)
        
        # embed captions using embedding layer
        embed = self.embed(cap)# shape: (batch_size, max_length, embed_dim)

        # concatenate image features with caption embeddings along the last dimension
        embed = torch.cat((y.unsqueeze(1).repeat(1, embed.shape[1], 1),embed), dim =2 )# shape: (batch_size, max_length, hidden_size*2 + embed_dim)
        
        # pass inputs through LSTM layer
        hiddens , _ = self.lstm(embed)# shape: (batch_size, max_length, hidden_size*2)

        printf("hiddens, embed: ", hiddens.shape, embed.shape)
        # pass outputs through transformer decoder layer
        hiddens = self.transformer_decoder(hiddens, embed) # shape: (batch_size, max_length, hidden_size*2)

        # pass outputs through linear layer to get logits for vocabulary
        outputs = self.linearDec(hiddens)# shape: (batch_size, max_length, vocab_size)
        return outputs


class Architecture3(nn.Module):
    def __init__(self,  hidden_size,embedding_size , no_layers, vocab_size, max_lengths = 50, applySoftmax=False):
        super(Architecture3, self).__init__()
        
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

                inp = torch.cat((img_features, y), dim=-1)
                hiddens , states = self.decoder.lstm(inp, states)
                hiddens = self.decoder.transformer_decoder(hiddens, inp)
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
    
