# import some packages you need here
import torch
import torch.nn.functional as F
import numpy as np
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature, char_to_idx, idx_to_char, min_length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

    Returns:
        samples: generated characters
    """

    # write your codes here
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_seq = torch.tensor([char_to_idx[char] for char in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
        # torch.tensor(...) converts the list of indices into a PyTorch tensor
        # dtype=torch.long specifies that the tensor should have a data type of long, which is typically used for indices.
        # .unsqueeze(0) adds a new dimension at the 0th position (i.e., the beginning of the tensor's shape). 
        # This is often done to add a batch dimension, converting the tensor from a 1D tensor (a single sequence of indices) to a 2D tensor (a batch with a single sequence).

    hidden = model.init_hidden(1) # batchsize = 1

    #samples = seed_characters
    samples = ''

    with torch.no_grad():
        while len(samples) < min_length:
            output, hidden = model(input_seq, hidden)
            output = output[:, -1, :] / temperature
                 # : selects all batches.
                 # -1 selects the last time step in each sequence.
                 # : selects all features.
            probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
                # This conversion is to interact with libraries or code that expect NumPy arrays instead of PyTorch tensors.
            next_char_idx = np.random.choice(len(probabilities), p=probabilities)
                # choose the charater index that has the highest probability
            next_char = idx_to_char[next_char_idx]
            
            samples += next_char
            
            # Prepare input for the next step
            input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

    return samples

if __name__ == '__main__':
    dataset = Shakespeare(input_file='./datasets/shakespeare_train.txt')
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    # Example model loading (Replace with actual code to load your best model)
    model_path = 'trained_lstm_model.pth'
    model_type = 'LSTM'  # or 'RNN'
    
    if model_type == 'LSTM':
        model = CharLSTM(len(char_to_idx), hidden_size=128, num_layers=2)
    else:
        model = CharRNN(len(char_to_idx), hidden_size=128, num_layers=2)
    
    model.load_state_dict(torch.load(model_path)) # use the best model

    seed_characters_list = [
        "He cannot temperately transport his honours", # next line: From where he should begin and end, but will
        "Were he to stand for consul, never would he", # next line: Appear i' the market-place nor on him put \n The napless vesture of humility;
        "We have power in ourselves to do it,", # next line: him manifests the true knowledge he has in their \n disposition; and out of his noble carelessness lets \n them plainly see'
        "Good night, good night! parting is such sweet sorrow,", # next line: but it is a\n power that we have no power to do; for if he show us\n his wounds and tell us his deeds, we are to put our
        "And with thy scorns drew'st rivers from his eyes" # next line: And then, to dry them, gavest the duke a clout
    ]

    # seed_characters_list = [
    #     "Hi, how are you?",
    #     "Where are you from?",
    #     "I think I am depressed by the results of the final exam",
    #     "I watched TV show last night, but it made me think that I should do homework afterwards",
    #     "Are you really sure that Michael Jordan would win the Final Championships?"
    # ]

    temperatures = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5] 

    for temperature in temperatures:
        print("at temperature ", temperature)
        for i, seed_characters in enumerate(seed_characters_list):
            generated_text = generate(model, seed_characters, temperature, char_to_idx, idx_to_char)
            
            print(f"Sample {i+1}")
            print('original characters: ',seed_characters)
            print('generated_text: ', generated_text)

            print("\n")

        print("\n" + "-"*50 + "\n")
