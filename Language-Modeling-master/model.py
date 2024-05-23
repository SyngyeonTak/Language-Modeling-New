import torch.nn as nn
import torch

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):

        # write your codes here
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):

        # write your codes here
        x = self.embedding(input)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(next(self.parameters()).device)

        return initial_hidden


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):

        # write your codes here
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)


    def forward(self, input, hidden):

        # write your codes here
        x = self.embedding(input)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)

        return output, hidden

    def init_hidden(self, batch_size):

        # write your codes here
        initial_hidden = (torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(next(self.parameters()).device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(next(self.parameters()).device))


        return initial_hidden