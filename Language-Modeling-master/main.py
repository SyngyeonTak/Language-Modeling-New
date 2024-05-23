import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt

# import some packages you need here


def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    model.train()
    total_loss = 0

    for batch in trn_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        hidden = model.init_hidden(inputs.size(0))

        outputs, _ = model(inputs, hidden)

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            # view function is used to reshape tensors. 
            # It allows you to change the dimensions of a tensor without changing its data. The view function is analogous to the reshape function in NumPy.

        loss.backward()            
        optimizer.step()

        total_loss += loss.item()


    trn_loss = total_loss / len(trn_loader)

    return trn_loss

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0))

            outputs, _ = model(inputs, hidden)

            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            total_loss += loss.item()

    val_loss = total_loss / len(val_loader)    

    return val_loss


def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation. 
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    batch_size = 64
    seq_length = 30
    hidden_size = 128
    num_layers = 2
    lr = 0.002
    epochs = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = Shakespeare(input_file='./datasets/shakespeare_train.txt')
    dataset_size = len(dataset) # return self.num_chunks - 1 (the number of input sequence)
    indices = list(range(dataset_size)) # [0 ~ dataset_size]
    split = int(np.floor(0.2 * dataset_size)) # training 0.8, validation 0.2

    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    trn_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    rnn_model = CharRNN(vocab_size=len(dataset.char_to_idx), hidden_size=hidden_size, num_layers=num_layers).to(device)
    lstm_model = CharLSTM(vocab_size=len(dataset.char_to_idx), hidden_size=hidden_size, num_layers=num_layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lr)

    rnn_trn_losses, rnn_val_losses = [], []
    lstm_trn_losses, lstm_val_losses = [], []

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train RNN model
        rnn_trn_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        rnn_trn_losses.append(rnn_trn_loss)
        rnn_val_losses.append(rnn_val_loss)
        
        # Train LSTM model
        lstm_trn_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        lstm_trn_losses.append(lstm_trn_loss)
        lstm_val_losses.append(lstm_val_loss)
        
        print(f"RNN - Training Loss: {rnn_trn_loss:.4f}, Validation Loss: {rnn_val_loss:.4f}")
        print(f"LSTM - Training Loss: {lstm_trn_loss:.4f}, Validation Loss: {lstm_val_loss:.4f}")


    model_path = 'trained_lstm_model.pth'
    torch.save(lstm_model.state_dict(), model_path) # this code is for saving the model_path to save the best performing model(LSTM)
    
    # Plot the loss values
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_trn_losses, label='RNN Training Loss')
    plt.plot(rnn_val_losses, label='RNN Validation Loss')
    plt.plot(lstm_trn_losses, label='LSTM Training Loss')
    plt.plot(lstm_val_losses, label='LSTM Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


if __name__ == '__main__':
    main()