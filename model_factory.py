from LSTM_model import ResNetLSTM
from Architecture2 import Architecture2
from TransformerArchitecture import Architecture3
from VanillaRNN import ResNetRNN

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    no_layers = config_data['model']['no_layers']
    vocab_size = len(vocab)

    # You may add more parameters if you want
    
    if model_type == "LSTM": ## Conditional clause based on model type
        model = ResNetLSTM(hidden_size, embedding_size, no_layers, vocab_size, 50)
    
    elif model_type == "Model2":
        image_embedding_size = config_data['model']['image_embedding_size']
        model = Architecture2(hidden_size,  image_embedding_size, embedding_size, no_layers, vocab_size, 50)

    elif model_type == "Model3":
        model = Architecture3(hidden_size, embedding_size, no_layers, vocab_size, 50)

    elif model_type == "RNN":
        model = ResNetRNN(hidden_size, embedding_size, no_layers, vocab_size, 50)

    else:
        raise NotImplementedError("Model Not Implemented")
    return model



        
                

                
                
            
    
    
    
    
    
    
    
   
